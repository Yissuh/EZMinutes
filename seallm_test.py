import ollama
from datetime import date
from language_detect import detect_language


class TranscriptProcessor:
    def __init__(self, transcript: str, agenda_items: list[str], model_name: str):
        self.transcript = transcript
        self.agenda_items = agenda_items
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
        # Build the appropriate prompt based on the detected language
        if self.language_detected == "TL":
            minutes_prompt = self._build_prompt_tagalog()  # Use the Tagalog prompt
        else:
            minutes_prompt = self._build_prompt()  # Use the default prompt (English)

        try:
            # Call the Ollama model with the selected prompt
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": minutes_prompt}]
            )

            # Extract the content field from the response
            if hasattr(response, 'message') and hasattr(response.message, 'content'):
                self.minutes = response.message.content.strip()
            else:
                raise ValueError("No 'content' field found in the response.")

            if not self.minutes:
                print("Warning: The model returned an empty content field.")
                self.minutes = "No meeting minutes could be generated. Please check the input data."
        except Exception as e:
            print(f"Error during Ollama chat: {e}")
            self.minutes = "Error occurred while generating meeting minutes."


    def _build_prompt_tagalog(self):
        """
        Build the prompt for generating meeting minutes in Tagalog.
        """
        agenda_section = "\n".join([f"- {agenda.strip()}" for agenda in self.agenda_items])

        return f"""
            Gumawa ng Minutes ng Pulong:

            Transcript: {self.transcript}
            
            Mga Agenda:
            {agenda_section}

            Mga Instruksyon:
            - Kunin ang mga mahahalagang detalye para sa bawat agenda.
            - Banggitin kung sino ang nagsabi ng bawat punto (hal. "SPEAKER 0").
            - Ayusin ang mga tala sa malinaw at madaling basahing format na may headings at bullet points.
            - Tiyakin na nakapokus lamang sa pinakamahalaga at aksyonableng impormasyon.
            - Gumamit ng ikatlong panauhan at tukuyin ang mga kalahok gamit ang kanilang identifier (hal. "SPEAKER 0").
            - Siguraduhing walang detalye ang inuulit at ang mga punto ay nauugnay lamang sa kaukulang agenda.
            - Huwag magdagdag ng impormasyong wala sa transcript.

            Format ng Output:
            Minutes ng Pulong para sa {date.today().strftime('%B %d, %Y')}
            [Pangalan ng Agenda]
            Mga Pangunahing Punto:
            [Punto na sinabi ni SPEAKER]
            Mga Aksyon/Desisyon:
            [Aksyon o desisyon ni SPEAKER]
            Mga Susunod na Hakbang:
            [Mga detalye para sa susunod na hakbang o "Walang binanggit na follow-up meeting o checkpoint sa transcript."]
            """


    def _build_prompt(self):
        """
        Build the prompt for generating meeting minutes.
        """
        agenda_section = "\n".join([f"- {agenda.strip()}" for agenda in self.agenda_items])

        return f"""
        Generate Meeting Minutes:

        Transcript: {self.transcript}
        
        Agendas:
        {agenda_section}

        Instructions:
        - Extract the key details for each agenda item.
        - Specify which speaker said each point.
        - Organize the notes in a clear, easy-to-scan format with headings and bullet points.
        - Focus on capturing the most important and actionable information.
        - Use third person and refer to participants by their identifiers (e.g., "SPEAKER 0").
        - Ensure no detail is repeated and all points strictly relate to the corresponding agenda item.
        - Do not add information not explicitly present in the transcript.

        Strictly follow the Output Format:
        Meeting Minutes for {date.today().strftime('%B %d, %Y')}
        [Agenda Item]
        Key Points:
        [Point stated by SPEAKER]
        Actions/Decisions:
        [Action or decision by SPEAKER]
        Follow-Up:
        [Follow-up details or "A follow-up meeting or checkpoint was not explicitly mentioned in the transcript."]
        """

    def save_minutes(self, output_file_path: str):
        """
        Save the generated meeting minutes to a text file.
        """
        if not self.minutes:
            raise ValueError("Meeting minutes have not been generated yet.")

        with open(output_file_path, "w", encoding="utf-8") as file:
            file.write(self.minutes)
            print(f"Minutes of the meeting have been saved to {output_file_path}")


class MeetingProcessorApp:
    def __init__(self, transcript_file_path: str, agenda_file_path: str, output_file_path: str, model_name: str):
        self.transcript_file_path = transcript_file_path
        self.output_file_path = output_file_path
        self.agenda_file_path = agenda_file_path
        self.model_name = model_name
        self.transcript = self._load_transcript()
        self.agenda_items = self._load_agenda()
        self.processor = TranscriptProcessor(self.transcript, self.agenda_items, self.model_name)

    def _load_transcript(self):
        """
        Load the transcript from the given file.
        """
        with open(self.transcript_file_path, "r", encoding="utf-8") as file:
            return file.read()

    def _load_agenda(self):
        """
        Load the agenda items from the file.
        """
        with open(self.agenda_file_path, "r", encoding="utf-8") as file:
            return [line.strip() for line in file.readlines()]

    def process_meeting(self):
        """
        Process the meeting by detecting language, generating minutes, and saving them.
        """
        self.processor.detect_language()    # Detect transcript language
        self.processor.generate_minutes()  # Generate meeting minutes
        self.processor.save_minutes(self.output_file_path)  # Save the minutes to a file


# Main entry point
def main():
    transcript_file_path = "testfiles/weekly_sample.txt"
    agenda_file_path = "testfiles/test_agenda.txt"
    output_file_path = "minutes_meeting.txt"
    model_name = "hf.co/QuantFactory/SeaLLMs-v3-7B-Chat-GGUF:Q5_K_S"

    app = MeetingProcessorApp(transcript_file_path, agenda_file_path, output_file_path, model_name)
    app.process_meeting()


if __name__ == "__main__":
    main()
