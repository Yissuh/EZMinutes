import ollama
from datetime import date
from language_detect import detect_language
import tiktoken

class TranscriptProcessor:
    def __init__(self, transcript: str, agenda_items: list[str], model_name: str = 'llama3.1:latest'):
        self.transcript = transcript
        self.agenda_items  = agenda_items
        self.model_name = model_name
        self.language_detected = None
        self.minutes = ""
        self.token_limit = 8192
        self.context_overlap = 100  # Number of tokens to overlap between chunks

    def detect_language(self):
        """
        Detect the language of the transcript.
        """
        self.language_detected = detect_language(self.transcript)
        print(f"Detected Language: {self.language_detected}")
    
    def generate_minutes(self):
        """
        Generate meeting minutes using the transcript and the model.
        """
        transcript_chunks = self._split_transcript(self.transcript)
        all_minutes = []

        for chunk in transcript_chunks:
            minutes_prompt = self._build_prompt(chunk)
            response = ollama.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': minutes_prompt}]
            )
            all_minutes.append(response['message']['content'])

        self.minutes = self._merge_minutes(all_minutes)
        
    def _split_transcript(self, transcript):
        """
        Split the transcript into chunks within the token limit, with overlapping context.
        """
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(transcript)
        chunks = []

        for i in range(0, len(tokens), self.token_limit - self.context_overlap):
            chunk_tokens = tokens[i:i + self.token_limit]
            chunk_text = encoding.decode(chunk_tokens)
            chunks.append(chunk_text)

        return chunks

    def _build_prompt(self, transcript_chunk):
        """
        Build the prompt for generating meeting minutes.
        """
        agenda_section = "\n".join([f"- {agenda.strip()}" for agenda in self.agenda_items])

        return f"""
        Prompt for Generating Simple Meeting Minutes
        From the provided {transcript_chunk}, extract the key details for each of the {agenda_section} and specify which speaker said each point. Ensure the following:

        Organize the notes in a clear, easy-to-scan format with headings and bullet points.
        Focus on capturing the most important and actionable information from the transcript.
        Write the minutes in third person, referring to each participant by their speaker identifier (e.g., "SPEAKER 0", "SPEAKER 1") or by name if available in the transcript.
        Avoid repeating any detail or statement. Ensure each point is mentioned only once and is strictly related to the corresponding agenda item.
        Do not add any information or context that is not explicitly present in the transcript.
        
        Generate the minutes strictly in this Output Format:

        Meeting Minutes for [DATE]
        [Agenda Item]
        Key Points:
        [Point stated by SPEAKER]

        Then extract these:
        Actions/Decisions:
        [Action or decision by SPEAKER]
        
        Follow-Up:
        [Follow-up details or "A follow-up meeting or checkpoint was not explicitly mentioned in the transcript."]
        """

    def _merge_minutes(self, all_minutes):
        """
        Merge the minutes from all chunks, removing redundancy.
        """
        merged_minutes = []
        seen_points = set()

        for minutes in all_minutes:
            lines = minutes.split('\n')
            for line in lines:
                if line not in seen_points:
                    merged_minutes.append(line)
                    seen_points.add(line)

        return "\n".join(merged_minutes)

    def save_minutes(self, output_file_path: str):
        """
        Save the generated meeting minutes to a text file.
        """
        if not self.minutes:
            raise ValueError("Meeting minutes have not been generated yet.")

        # Save the response to the file
        with open(output_file_path, 'w', encoding='utf-8') as file:
            file.write(self.minutes.replace("**", ""))  # Save the response to the file
            print(f"Minutes of the meeting have been saved to {output_file_path}")


class MeetingProcessorApp:
    def __init__(self, transcript_file_path: str, agenda_file_path: str, output_file_path: str):
        self.transcript_file_path = transcript_file_path
        self.output_file_path = output_file_path
        self.agenda_file_path = agenda_file_path
        self.transcript = self._load_transcript()
        self.agenda_items = self._load_agenda()
        self.processor = TranscriptProcessor(self.transcript, self.agenda_items)


    def _load_transcript(self):
        """
        Load the transcript from the given file.
        """
        with open(self.transcript_file_path, 'r') as file:
            return file.read()
        
    def _load_agenda(self):
        """
        Load the agenda items from the file.
        """
        with open(self.agenda_file_path, 'r') as file:
            return file.readlines()
        
    def process_meeting(self):
        """
        Process the meeting by detecting language, generating minutes, and saving them.
        """
        self.processor.detect_language()  # Detect language
        self.processor.generate_minutes()  # Generate meeting minutes
        self.processor.save_minutes(self.output_file_path)  # Save the minutes to a file


# Main entry point
def main():
    transcript_file_path = "testfiles/techguild.txt"
    output_file_path = "minutes.txt"
    agenda_file_path = "testfiles/empty.txt"

    app = MeetingProcessorApp(transcript_file_path, agenda_file_path, output_file_path)
    app.process_meeting()

if __name__ == "__main__":
    main()