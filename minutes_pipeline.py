import json
import os
import ollama
import time
import tiktoken
import concurrent.futures
from functools import partial

def extract_meeting_minutes(transcript_path, agenda_path=None, output_language="English", temperature=0.5, max_workers=None):
    """
    Extract meeting minutes from a transcript file using Quantized Ollama's Llama 3.1 8b Instruct
    with parallel processing of transcript chunks
    
    Args:
        transcript_path (str): Path to the transcript file
        agenda_path (str, optional): Path to the agenda file
        output_language (str): Language for the output (default: English)
        temperature (float): Temperature parameter for controlling randomness (0.0-1.0)
        max_workers (int, optional): Maximum number of worker processes (None = auto)
    
    Returns:
        dict: Structured meeting minutes
    """
    # Read transcript file
    with open(transcript_path, 'r') as f:
        transcript = f.read()
    
    # Process transcript if it's long
    transcript_chunks = chunk_transcript(transcript)
    print(f"Transcript split into {len(transcript_chunks)} chunks for processing")
    
    # Check if agenda file exists and is not empty
    agenda_items = []
    if agenda_path and os.path.exists(agenda_path) and os.path.getsize(agenda_path) > 0:
        with open(agenda_path, 'r') as f:
            agenda_items = [line.strip() for line in f.readlines() if line.strip()]
    else:
        # Dynamically identify agenda items in parallel
        agenda_items = identify_agenda_items_from_chunks_parallel(transcript_chunks, output_language, temperature, max_workers)
    
    # Generate meeting overview in parallel
    overview = generate_overview_from_chunks_parallel(transcript_chunks, agenda_items, output_language, temperature, max_workers)
    
    # Extract discussion points for each agenda item in parallel
    discussion_points = extract_discussion_points_from_chunks_parallel(transcript_chunks, agenda_items, output_language, temperature, max_workers)
    
    # Extract action items in parallel
    action_items = extract_action_items_from_chunks_parallel(transcript_chunks, output_language, temperature, max_workers)
    
    # Combine all results into a single structure
    meeting_minutes = {
        "meeting_overview": overview,
        "agenda_items": agenda_items,
        "discussion_points": discussion_points,
        "action_items": action_items
    }
    
    return meeting_minutes

def chunk_transcript(transcript, max_chunk_size=4096, overlap_size=500):
    """
    Split a long transcript into manageable chunks with overlapping context
    using tiktoken for accurate token counting
    
    Args:
        transcript (str): The full transcript text
        max_chunk_size (int): Maximum size of each chunk in tokens
        overlap_size (int): Size of the overlap between chunks in tokens
        
    Returns:
        list: List of transcript chunks
    """
    
    # Initialize the tiktoken encoder (using cl100k_base which is used by many newer models)
    enc = tiktoken.get_encoding("cl100k_base")
    
    # If transcript is shorter than max_chunk_size, return it as a single chunk
    tokens = enc.encode(transcript)
    print(len(tokens))
    if len(tokens) <= max_chunk_size:
        return [transcript]
    
    # Split transcript by lines
    lines = transcript.split('\n')
    
    chunks = []
    current_chunk_lines = []
    current_chunk_tokens = []
    overlap_buffer_lines = []
    overlap_buffer_tokens = []
    
    for line in lines:
        line_tokens = enc.encode(line + '\n')  # Include newline in token count
        
        # If adding this line would exceed the max chunk size and we already have content
        if current_chunk_tokens and len(current_chunk_tokens) + len(line_tokens) > max_chunk_size:
            # Save the current chunk
            chunks.append('\n'.join(current_chunk_lines))
            
            # Set up the next chunk with overlap
            # Find speaker markers in the overlap buffer to ensure we start with a speaker
            overlap_text = '\n'.join(overlap_buffer_lines)
            
            # Initialize the next chunk with the overlap buffer
            current_chunk_lines = overlap_buffer_lines.copy()
            current_chunk_tokens = overlap_buffer_tokens.copy()
            
            # Reset overlap buffer for the next chunk
            overlap_buffer_lines = []
            overlap_buffer_tokens = []
        
        # Add the current line to the chunk
        current_chunk_lines.append(line)
        current_chunk_tokens.extend(line_tokens)
        
        # Maintain the overlap buffer with the most recent lines
        overlap_buffer_lines.append(line)
        overlap_buffer_tokens.extend(line_tokens)
        
        # Keep the overlap buffer at the appropriate size
        while len(overlap_buffer_tokens) > overlap_size:
            removed_line = overlap_buffer_lines.pop(0)
            removed_tokens = enc.encode(removed_line + '\n')
            overlap_buffer_tokens = overlap_buffer_tokens[len(removed_tokens):]
    
    # Add the last chunk if it's not empty
    if current_chunk_lines:
        chunks.append('\n'.join(current_chunk_lines))
    
    # Ensure each chunk starts with a speaker marker if possible
    for i in range(1, len(chunks)):
        if not chunks[i].strip().startswith("SPEAKER"):
            # Find the first speaker marker
            speaker_index = chunks[i].find("SPEAKER")
            if speaker_index > 0:
                # Move the text before the speaker marker to the previous chunk
                chunks[i-1] += chunks[i][:speaker_index]
                chunks[i] = chunks[i][speaker_index:]
    
    return chunks

# Functions for parallel processing of agenda items
def process_chunk_for_agenda(chunk, output_language, temperature, chunk_index, total_chunks):
    """Process a single chunk for agenda items"""
    print(f"Processing chunk {chunk_index+1}/{total_chunks} for agenda items")
    agenda_prompt = generate_dynamic_agenda_prompt(chunk, output_language)
    
    agenda_response = ollama.chat(
        model="llama-3.1-8b-q4:latest", 
        messages=[
            {"role": "system", "content": f"You are a helpful assistant that extracts meeting agenda items from transcripts. Always respond in {output_language} regardless of the input language."},
            {"role": "user", "content": agenda_prompt}
        ], 
        format="json",
        options={"temperature": temperature, "mirostat" : 2.0}
    )
    
    try:
        agenda_result = json.loads(agenda_response['message']['content'])
        return agenda_result.get('agenda_items', [])
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error processing chunk {chunk_index+1} for agenda items: {e}")
        print(f"Raw response: {agenda_response['message']['content']}")
        return []

def identify_agenda_items_from_chunks_parallel(transcript_chunks, output_language, temperature, max_workers=None):
    """
    Identify agenda items from transcript chunks in parallel
    
    Args:
        transcript_chunks (list): List of transcript text chunks
        output_language (str): Language for the output
        temperature (float): Temperature parameter for controlling randomness
        max_workers (int, optional): Maximum number of worker processes
        
    Returns:
        list: Identified agenda items
    """
    all_agenda_items = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a list of futures
        future_to_chunk = {
            executor.submit(
                process_chunk_for_agenda, 
                chunk, 
                output_language, 
                temperature, 
                i, 
                len(transcript_chunks)
            ): i for i, chunk in enumerate(transcript_chunks)
        }
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_chunk):
            try:
                chunk_agenda_items = future.result()
                all_agenda_items.extend(chunk_agenda_items)
            except Exception as e:
                chunk_index = future_to_chunk[future]
                print(f"Exception processing chunk {chunk_index+1} for agenda: {e}")
    
    # Consolidate and deduplicate agenda items
    consolidated_items = consolidate_agenda_items(all_agenda_items)
    return consolidated_items

def consolidate_agenda_items(agenda_items):
    """
    Consolidate and deduplicate agenda items
    
    Args:
        agenda_items (list): List of all extracted agenda items
        
    Returns:
        list: Consolidated list of unique agenda items
    """
    if not agenda_items:
        return []
    
    # Create a consolidated prompt to refine the agenda items
    consolidated_prompt = f"""
    I have extracted the following potential agenda items from different parts of a meeting transcript:
    {json.dumps(agenda_items)}
    
    Please consolidate these into a concise list of 3-7 main agenda items, removing duplicates and combining similar topics.
    
    Respond with a JSON object in the following format:
    {{
        "agenda_items": ["topic1", "topic2", "topic3"]
    }}
    """
    
    consolidation_response = ollama.chat(
        model="llama-3.1-8b-q4:latest", 
        messages=[
            {"role": "system", "content": "You are a helpful assistant that consolidates meeting agenda items."},
            {"role": "user", "content": consolidated_prompt}
        ], 
        format="json",
        options={"temperature": 0, "mirostat" : 2.0}  # Lower temperature for more deterministic results
    )
    
    try:
        consolidation_result = json.loads(consolidation_response['message']['content'])
        return consolidation_result.get('agenda_items', [])
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error consolidating agenda items: {e}")
        print(f"Raw response: {consolidation_response['message']['content']}")
        # Fall back to basic deduplication
        return list(set(agenda_items))

# Functions for parallel processing of meeting overview
def process_chunk_for_overview(chunk, agenda_items, output_language, temperature, chunk_index, total_chunks):
    """Process a single chunk for meeting overview"""
    print(f"Processing chunk {chunk_index+1}/{total_chunks} for overview")
    overview_prompt = generate_overview_prompt(chunk, agenda_items, output_language)
    
    overview_response = ollama.chat(
        model="llama-3.1-8b-q4:latest", 
        messages=[
            {"role": "system", "content": f"You are a helpful assistant that creates concise meeting overviews. Always respond in {output_language} regardless of the input language."},
            {"role": "user", "content": overview_prompt}
        ], 
        format="json",
        options={"temperature": temperature, "mirostat" : 2.0}
    )
    
    try:
        overview_result = json.loads(overview_response['message']['content'])
        return overview_result.get('meeting_overview', '')
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error processing chunk {chunk_index+1} for overview: {e}")
        print(f"Raw response: {overview_response['message']['content']}")
        return ""

def generate_overview_from_chunks_parallel(transcript_chunks, agenda_items, output_language, temperature, max_workers=None):
    """
    Generate meeting overview from transcript chunks in parallel
    
    Args:
        transcript_chunks (list): List of transcript text chunks
        agenda_items (list): List of identified agenda items
        output_language (str): Language for the output
        temperature (float): Temperature parameter for controlling randomness
        max_workers (int, optional): Maximum number of worker processes
        
    Returns:
        str: Meeting overview
    """
    chunk_summaries = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a list of futures
        future_to_chunk = {
            executor.submit(
                process_chunk_for_overview, 
                chunk, 
                agenda_items, 
                output_language, 
                temperature, 
                i, 
                len(transcript_chunks)
            ): i for i, chunk in enumerate(transcript_chunks)
        }
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_chunk):
            try:
                chunk_summary = future.result()
                if chunk_summary:
                    chunk_summaries.append(chunk_summary)
            except Exception as e:
                chunk_index = future_to_chunk[future]
                print(f"Exception processing chunk {chunk_index+1} for overview: {e}")
    
    # Combine chunk summaries into a final overview
    if chunk_summaries:
        if len(chunk_summaries) == 1:
            return chunk_summaries[0]
        else:
            return consolidate_overviews(chunk_summaries, agenda_items, output_language)
    else:
        return "No overview could be generated from the transcript."

def consolidate_overviews(chunk_summaries, agenda_items, output_language):
    """
    Consolidate multiple chunk overviews into a single cohesive overview
    
    Args:
        chunk_summaries (list): List of overview summaries from each chunk
        agenda_items (list): List of identified agenda items
        output_language (str): Language for the output
        
    Returns:
        str: Consolidated meeting overview
    """
    # Create a consolidated prompt
    consolidated_prompt = f"""
    I have generated the following overview summaries from different parts of a meeting transcript:
    {json.dumps(chunk_summaries)}
    
    The meeting discussed these agenda items:
    {json.dumps(agenda_items)}
    
    Please consolidate these summaries into a single coherent overview paragraph that captures the essence of the meeting.
    
    Respond with a JSON object in the following format:
    {{
        "meeting_overview": "Your consolidated meeting overview here."
    }}
    """
    
    consolidation_response = ollama.chat(
        model="llama-3.1-8b-q4:latest", 
        messages=[
            {"role": "system", "content": f"You are a helpful assistant that consolidates meeting overviews. Always respond in {output_language} regardless of the input language."},
            {"role": "user", "content": consolidated_prompt}
        ], 
        format="json",
        options={"temperature": 0, "mirostat" : 2.0}  # Lower temperature for more deterministic results
    )
    
    try:
        consolidation_result = json.loads(consolidation_response['message']['content'])
        return consolidation_result.get('meeting_overview', '')
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error consolidating overviews: {e}")
        print(f"Raw response: {consolidation_response['message']['content']}")
        # Fall back to concatenation
        return " ".join(chunk_summaries)

# Functions for parallel processing of discussion points
def process_chunk_for_discussion(chunk, agenda_items, output_language, temperature, chunk_index, total_chunks):
    """Process a single chunk for discussion points"""
    print(f"Processing chunk {chunk_index+1}/{total_chunks} for discussion points")
    discussion_prompt = generate_discussion_prompt(chunk, agenda_items, output_language)
    
    discussion_response = ollama.chat(
        model="llama-3.1-8b-q4:latest", 
        messages=[
            {"role": "system", "content": f"You are a helpful assistant that extracts the top key points from meeting transcripts. Always respond in {output_language} regardless of the input language."},
            {"role": "user", "content": discussion_prompt}
        ], 
        format="json",
        options={"temperature": temperature, "mirostat" : 2.0}
    )
    
    try:
        discussion_result = json.loads(discussion_response['message']['content'])
        return discussion_result.get('discussion_points', [])
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error processing chunk {chunk_index+1} for discussion points: {e}")
        print(f"Raw response: {discussion_response['message']['content']}")
        return []

def extract_discussion_points_from_chunks_parallel(transcript_chunks, agenda_items, output_language, temperature, max_workers=None):
    """
    Extract discussion points from transcript chunks in parallel
    
    Args:
        transcript_chunks (list): List of transcript text chunks
        agenda_items (list): List of identified agenda items
        output_language (str): Language for the output
        temperature (float): Temperature parameter for controlling randomness
        max_workers (int, optional): Maximum number of worker processes
        
    Returns:
        list: Discussion points organized by agenda item
    """
    all_discussion_points = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a list of futures
        future_to_chunk = {
            executor.submit(
                process_chunk_for_discussion, 
                chunk, 
                agenda_items, 
                output_language, 
                temperature, 
                i, 
                len(transcript_chunks)
            ): i for i, chunk in enumerate(transcript_chunks)
        }
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_chunk):
            try:
                chunk_points = future.result()
                all_discussion_points.extend(chunk_points)
            except Exception as e:
                chunk_index = future_to_chunk[future]
                print(f"Exception processing chunk {chunk_index+1} for discussion points: {e}")
    
    # Consolidate discussion points by agenda item
    return consolidate_discussion_points(all_discussion_points, agenda_items)

def consolidate_discussion_points(all_points, agenda_items):
    """
    Consolidate discussion points by agenda item
    
    Args:
        all_points (list): List of all extracted discussion points from all chunks
        agenda_items (list): List of identified agenda items
        
    Returns:
        list: Consolidated discussion points organized by agenda item
    """
    # Create a dictionary to store points for each agenda item
    agenda_point_map = {item: [] for item in agenda_items}
    
    # Collect all points for each agenda item
    for point_group in all_points:
        agenda_item = point_group.get('agenda_item', '')
        points = point_group.get('points', [])
        
        # Find the best matching agenda item
        best_match = find_best_matching_agenda_item(agenda_item, agenda_items)
        if best_match:
            agenda_point_map[best_match].extend(points)
    
    # Format the consolidated points
    consolidated_points = []
    for agenda_item, points in agenda_point_map.items():
        # Remove duplicates
        unique_points = []
        seen_points = set()
        
        for point in points:
            point_text = point.get('point', '').lower()
            if point_text and point_text not in seen_points:
                seen_points.add(point_text)
                unique_points.append(point)
        
        if unique_points:
            consolidated_points.append({
                "agenda_item": agenda_item,
                "points": unique_points
            })
    
    return consolidated_points

def find_best_matching_agenda_item(source_item, agenda_items):
    """
    Find the best matching agenda item from the list
    
    Args:
        source_item (str): The source agenda item to match
        agenda_items (list): List of target agenda items
        
    Returns:
        str: The best matching agenda item
    """
    if source_item in agenda_items:
        return source_item
    
    # Simple matching - find the agenda item with most word overlap
    source_words = set(source_item.lower().split())
    best_match = None
    best_overlap = 0
    
    for item in agenda_items:
        item_words = set(item.lower().split())
        overlap = len(source_words.intersection(item_words))
        if overlap > best_overlap:
            best_overlap = overlap
            best_match = item
    
    # If no good match found, use the first agenda item
    if not best_match and agenda_items:
        best_match = agenda_items[0]
    
    return best_match

# Functions for parallel processing of action items
def process_chunk_for_action(chunk, output_language, temperature, chunk_index, total_chunks):
    """Process a single chunk for action items"""
    print(f"Processing chunk {chunk_index+1}/{total_chunks} for action items")
    action_prompt = generate_action_prompt(chunk, output_language)
    
    action_response = ollama.chat(
        model="llama-3.1-8b-q4:latest", 
        messages=[
            {"role": "system", "content": f"You are a helpful assistant that extracts future TO-DO action items from meeting transcripts. Always respond in {output_language} regardless of the input language."},
            {"role": "user", "content": action_prompt}
        ], 
        format="json",
        options={"temperature": temperature, "mirostat" : 2.0}
    )
    
    try:
        action_result = json.loads(action_response['message']['content'])
        return action_result.get('action_items', [])
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error processing chunk {chunk_index+1} for action items: {e}")
        print(f"Raw response: {action_response['message']['content']}")
        return []

def extract_action_items_from_chunks_parallel(transcript_chunks, output_language, temperature, max_workers=None):
    """
    Extract action items from transcript chunks in parallel
    
    Args:
        transcript_chunks (list): List of transcript text chunks
        output_language (str): Language for the output
        temperature (float): Temperature parameter for controlling randomness
        max_workers (int, optional): Maximum number of worker processes
        
    Returns:
        list: Extracted action items
    """
    all_action_items = []

    
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a list of futures
        future_to_chunk = {
            executor.submit(
                process_chunk_for_action, 
                chunk, 
                output_language, 
                temperature, 
                i, 
                len(transcript_chunks)
            ): i for i, chunk in enumerate(transcript_chunks)
        }
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_chunk):
            try:
                chunk_actions = future.result()
                all_action_items.extend(chunk_actions)
            except Exception as e:
                chunk_index = future_to_chunk[future]
                print(f"Exception processing chunk {chunk_index+1} for action items: {e}")
          
    
    # Deduplicate action items
    return deduplicate_action_items(all_action_items)

def deduplicate_action_items(action_items):
    """
    Deduplicate action items
    
    Args:
        action_items (list): List of all extracted action items
        
    Returns:
        list: Deduplicated action items
    """
    unique_actions = []
    seen_actions = set()
    
    for item in action_items:
        # Create a signature for the action item
        assignee = item.get('assignee', '').lower()
        action = item.get('action', '').lower()
        signature = f"{assignee}:{action}"
        
        if signature not in seen_actions:
            seen_actions.add(signature)
            unique_actions.append(item)
    
    return unique_actions


# The prompt generation functions remain the same
def generate_dynamic_agenda_prompt(transcript, output_language="English"):
    """Generate a prompt to identify agenda items dynamically"""
    return f"""
    Based on the following meeting transcript, identify the TOP meaningful agendas/meeting topics dynamically.
    Make sure to focus on the most important topics, without redundant or similar context.
    
    Transcript:
    {transcript}
    
    Respond with a JSON object in the following format:
    {{
        "agenda_items": ["topic1", "topic2", "topic3"]
    }}
    
    Include only the most significant topics discussed in the meeting.
    Your response must be in {output_language} regardless of the language in the transcript.
    """

def generate_overview_prompt(transcript, agenda_items, output_language="English"):
    """Generate a prompt to create a meeting overview"""
    return f"""
    Based on the following meeting transcript and agenda items, write a 1 paragraph short and concise meeting overview.
    The overview should capture the essence of the meeting without delving too much into details.
    
    Transcript:
    {transcript}
    
    Agenda Items:
    {json.dumps(agenda_items)}
    
    Respond with a JSON object in the following format:
    {{
        "meeting_overview": "Your concise meeting overview here."
    }}
    
    Your response must be in {output_language} regardless of the language in the transcript.
    """

def generate_discussion_prompt(transcript, agenda_items, output_language="English"):
    """Generate a prompt to extract key points for each agenda item"""
    return f"""
    Based on the following meeting transcript and agenda items, extract and summarize the top most important key points
    for each agenda item. Each key point should be:
    1. A concise summary (not a verbatim quote)
    2. Written in third person point of view
    3. Include the speaker's ID
    4. Capture the essence of what was said without being too detailed
    
    Do NOT include direct quotes. Instead, paraphrase and summarize the key ideas.
    
    Transcript:
    {transcript}
    
    Agenda Items:
    {json.dumps(agenda_items)}
    
    Respond with a JSON object in the following format:
    {{
        "discussion_points": [
            {{
                "agenda_item": "topic1",
                "points": [
                    {{
                        "speaker": "SPEAKER X", 
                        "point": "Concise summary of the key point in third person"
                    }},
                    {{
                        "speaker": "SPEAKER Y", 
                        "point": "Concise summary of the key point in third person"
                    }}
                ]
            }},
            {{
                "agenda_item": "topic2",
                "points": [
                    {{
                        "speaker": "SPEAKER Z", 
                        "point": "Concise summary of the key point in third person"
                    }}
                ]
            }}
        ]
    }}
    
    Important guidelines:
    - Each point should be 1-2 sentences maximum
    - Focus on the main idea, not every detail
    - Combine related ideas from the same speaker
    - Remove filler words and repetitions
    - Maintain the original meaning but express it more concisely
    
    Your response must be in {output_language} regardless of the language in the transcript.
    """

def generate_action_prompt(transcript, output_language="English"):
    """Generate a prompt to extract action items"""
    return f"""
    Based on the following meeting transcript, extract all to-do action items.
    A to-do action item is a future task that someone has committed to doing.
    
    Transcript:
    {transcript}
    
    Respond with a JSON object in the following format:
    {{
        "action_items": [
            {{"assignee": "Speaker X", "action": "Description of the action to be taken"}},
            {{"assignee": "Speaker Y", "action": "Description of the action to be taken"}}
        ]
    }}
    
    Include only clear future action items where someone has committed to doing something specific.
    Your response must be in {output_language} regardless of the language in the transcript.
    """

def save_minutes_to_file(minutes, output_path):
    """Save meeting minutes to a file"""
    with open(output_path, 'w') as f:
        json.dump(minutes, f, indent=2)

def main():
    # Example usage
    transcript_path = "testfiles/9406.txt"
    agenda_path = ""  # Optional
    output_path = "meeting_minutes_2.json"
    output_language = "English"
    temperature = 0.5# Lower temperature for more deterministic outputs
    max_workers = None  # Let Python decide the optimal number based on system
    
    start_time = time.time()
    minutes = extract_meeting_minutes(transcript_path, agenda_path, output_language, temperature, max_workers)
    save_minutes_to_file(minutes, output_path)
    
    print(f"Meeting minutes saved to {output_path}")
    elapsed_time = time.time() - start_time
    print(f"Total Time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()