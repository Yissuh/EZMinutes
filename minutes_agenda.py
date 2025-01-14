import ollama
from datetime import date
from language_detect import detect_language


class TranscriptProcessor:
    def __init__(self, transcript: str, agenda_items: list[str], model_name: str = 'llama3.1:latest'):
        self.transcript = transcript
        self.agenda_items  = agenda_items
        self.model_name = model_name
        self.language_detected = None
        self.minutes = ""

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
        minutes_prompt = self._build_prompt()

        # Call the Ollama model with the selected prompt
        response = ollama.chat(
            model=self.model_name,
            messages=[{'role': 'user', 'content': minutes_prompt}]
        )
        self.minutes = response['message']['content']
        
    def _build_prompt(self):
        """
        Build the prompt for generating meeting minutes.
        """
        agenda_section = "\n".join([f"- {agenda.strip()}" for agenda in self.agenda_items])

        return f"""
        I am preparing the meeting minutes for a meeting held on [{date.today().strftime('%Y-%m-%d')}]
        The following agenda is the main objective of the meeting: {agenda_section}

        Generate comprehensive meeting minutes from the provided transcript including the following:
        - Discussion Points: Detail the topics discussed, including any debates or alternate viewpoints, exactly as they appear in the transcript.
        - Decisions Made: Record all decisions, including who made them and the rationale, as stated in the transcript.
        - Action Items: Specify tasks assigned, responsible individuals, and deadlines. List each one with the assigned owner and due date in the format: "[Owner] suggested [action item]". Include only the action items directly mentioned in the transcript.
        - Data & Insights: Summarize any data presented or insights shared that influenced the meeting's course. Include relevant context or details about the meeting, such as the meeting type (e.g., sales call, project update), attendees based on the speaker, and the overall objective. Stick to what was mentioned in the transcript.
        - Follow-Up: Note any agreed-upon follow-up meetings or checkpoints. Only include follow-ups explicitly mentioned in the transcript. If there is no follow-ups, just state: "A follow-up meeting or checkpoint was not explicitly mentioned in the transcript"

        Organize the notes in a clear, easy-to-scan format with headings and bullet points. Focus on capturing the most important and actionable information from the transcript.
        Remember to write the minutes in third person, referring to each participant by their speaker identifier (e.g., 'SPEAKER 0', 'SPEAKER 1') or by name if available in the transcript. DO NOT add any information or context that is not explicitly present in the transcript.

        Assume the date of the meeting is today.
        Below is the transcript:

        {self.transcript}

        Generate the output strictly following this expected format for the meeting minutes:

        ### Meeting Minutes for {date.today().strftime('%Y-%m-%d')}

        #### Meeting Title:

        #### Discussion Points:

        #### Decisions Made:

        #### Action Items:

        #### Data & Insights:

        #### Follow-Up:
        """

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
    transcript_file_path = "testfiles/weekly_sample.txt"
    output_file_path = "minutes.txt"
    agenda_file_path = "testfiles/test_agenda.txt"

    app = MeetingProcessorApp(transcript_file_path, agenda_file_path, output_file_path)
    app.process_meeting()

if __name__ == "__main__":
    main()
