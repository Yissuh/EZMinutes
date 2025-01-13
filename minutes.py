import ollama
from datetime import date
from language_detect import detect_language

# Load the transcript
file_path = "testfiles/chair meeting.txt"
with open(file_path, 'r') as file:
    transcript = file.read()

# Define the meeting minutes prompt for Filipino
minutes_prompt_filipino = f"""
Ako ay naghahanda ng mga minutong tala para sa isang pagpupulong na naganap noong {date.today().strftime('%Y-%m-%d')}.
Siguraduhing sumagot sa Tagalog.
Mangyaring gumawa ng komprehensibong minutong tala mula sa ibinigay na transcript na kinabibilangan ng mga sumusunod:
- Mga Puntos ng Diskusyon: I-detalye ang mga paksang tinalakay, kasama na ang mga debate o alternatibong pananaw, eksaktong tulad ng pagkakalagay sa transcript.
- Mga Desisyon na Ginawa: I-record ang lahat ng desisyon, kasama na kung sino ang gumawa ng desisyon at ang kanilang mga dahilan, ayon sa transcript.
- Mga Gawaing Nakatalaga: Tukuyin ang mga gawain na itinalaga, mga responsableng tao, at mga deadline. Ilista ang bawat isa gamit ang format na "[May-ari] ay nagmungkahi ng [gawain]". Isama lamang ang mga gawain na direktang binanggit sa transcript.
- Data at Mga Insight: Ibuod ang anumang datos na ipinakita o mga pananaw na ibinahagi na naka-apekto sa daloy ng pagpupulong. Isama ang mga kaugnay na konteksto o detalye tungkol sa pagpupulong, tulad ng uri ng pagpupulong (hal., sales call, update ng proyekto), mga kalahok batay sa nagsalita, at ang kabuuang layunin. Manatili sa mga binanggit sa transcript.
- Mga Follow-Up: I-note ang anumang mga napagkasunduang follow-up na pagpupulong o mga checkpoint. Isama lamang ang mga follow-up na hayagang binanggit sa transcript.

Ayusin ang mga tala sa isang malinaw, madaling basahing format na may mga heading at bullet points. Ituon ang pansin sa pinakamahalaga at pinakamataas na aksyon mula sa transcript.
Tandaan na isulat ang mga minutong tala sa pangatlong panauhan, tinutukoy ang bawat kalahok gamit ang kanilang speaker identifier (hal., 'SPEAKER 0', 'SPEAKER 1') o pangalan kung ito ay available sa transcript. HUWAG magdagdag ng anumang impormasyon o konteksto na hindi hayagang nabanggit sa transcript.

Tanggapin na ang petsa ng pagpupulong ay ngayon araw.
Nasa ibaba ang transcript:

{transcript}

"""

# Define the meeting minutes prompt for English
minutes_prompt_english = f"""
I am preparing the meeting minutes for a meeting held on {date.today().strftime('%Y-%m-%d')}.

Please generate comprehensive meeting minutes from the provided transcript including the following:
- Discussion Points: Detail the topics discussed, including any debates or alternate viewpoints, exactly as they appear in the transcript.
- Decisions Made: Record all decisions, including who made them and the rationale, as stated in the transcript.
- Action Items: Specify tasks assigned, responsible individuals, and deadlines. List each one with the assigned owner and due date in the format: "[Owner] suggested [action item]". Include only the action items directly mentioned in the transcript.
- Data & Insights: Summarize any data presented or insights shared that influenced the meeting's course. Include relevant context or details about the meeting, such as the meeting type (e.g., sales call, project update), attendees based on the speaker, and the overall objective. Stick to what was mentioned in the transcript.
- Follow-Up: Note any agreed-upon follow-up meetings or checkpoints. Only include follow-ups explicitly mentioned in the transcript. If there is no follow-ups, just state: "A follow-up meeting or checkpoint was not explicitly mentioned in the transcript"

Organize the notes in a clear, easy-to-scan format with headings and bullet points. Focus on capturing the most important and actionable information from the transcript.
Remember to write the minutes in third person, referring to each participant by their speaker identifier (e.g., 'SPEAKER 0', 'SPEAKER 1') or by name if available in the transcript. DO NOT add any information or context that is not explicitly present in the transcript.

Assume the date of the meeting is today.
Below is the transcript:

{transcript}

Here is an example of the expected output format for the meeting minutes:

### Meeting Minutes for {date.today().strftime('%Y-%m-%d')}

#### Discussion Points:

#### Decisions Made:

#### Action Items:

#### Data & Insights:

#### Follow-Up:
"""



# Detect the language of the transcript
language_detected = detect_language(transcript)

# Debugging print to check the detected language
print(f"Detected Language: {language_detected}")

# Choose the appropriate prompt based on the detected language
if language_detected == 'tl':  # Tagalog
    print("Using Filipino prompt.")
    prompt = minutes_prompt_filipino
elif language_detected == 'en':  # English
    print("Using English prompt.")
    prompt = minutes_prompt_english
else:  # Mixed or unknown
    print("Using default English prompt (for mixed/unknown).")
    prompt = minutes_prompt_english

# Call the Ollama model with the selected prompt
desiredModel = 'llama3.1:latest'
questionToAsk = prompt

response = ollama.chat(
    model=desiredModel,
    messages=[{'role': 'user', 'content': questionToAsk}]
)

ollama_response = response['message']['content']

# Print the response without markdown (remove '**' from the output)
print("Model Response:\n", ollama_response.replace("**", ""))
# Export the output to a text file
output_file_path = "minutes.txt"
with open(output_file_path, 'w', encoding='utf-8') as file:
    file.write(ollama_response.replace("**", ""))  # Save the response to the file
    print(f"Minutes of the meeting have been saved to {output_file_path}")
