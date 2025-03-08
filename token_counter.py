import tiktoken
import os

def count_tokens(text, encoding_name="cl100k_base"):
    """
    Count the number of tokens in a text using the specified tokenizer.
    Default is cl100k_base which is used by Claude models.
    """
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        tokens = encoding.encode(text)
        return len(tokens)
    except Exception as e:
        print(f"Error in tokenization: {e}")
        return 0

def main():
    transcript_file_path = "testfiles/weekly_sample.txt"
    llama_token_limit = 8192
    
    # Check if file exists
    if not os.path.exists(transcript_file_path):
        print(f"Error: File not found at {transcript_file_path}")
        return
    
    # Read transcript
    try:
        with open(transcript_file_path, "r", encoding="utf-8") as f:
            transcript = f.read()
        
        # Count tokens
        token_count = count_tokens(transcript)
        print(f"Transcript Token Count: {token_count}")
        
        # Check if token count is acceptable for LLaMA 3.1 7B
        if token_count <= llama_token_limit:
            print("Token count is acceptable for LLaMA 3.1 7B.")
        else:
            print("Token count exceeds the limit for LLaMA 3.1 7B.")
    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    main()