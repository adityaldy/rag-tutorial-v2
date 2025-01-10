# Step 1: Import the required libraries
import PyPDF2
from transformers import pipeline

# Step 2: Define a function to check if the PDF file can be opened and extract text
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            # Check if the PDF has at least one page
            if len(reader.pages) > 0:
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text  # Return the extracted text
            else:
                return None
    except Exception as e:
        print(f"Failed to open PDF file: {e}")
        return None

# Step 3: Define a function to generate questions from the text
def generate_questions(text):
    # Use a model for question generation
    # question_generator = pipeline("text2text-generation", model="valhalla/t5-small-qa-qg-hl")
    question_generator = pipeline("text2text-generation", model="t5-small")
    sentences = text.split('. ')
    questions = []

    for sentence in sentences:
        if len(sentence.split()) > 5:  # Only take sufficiently long sentences
            generated_question = question_generator(f"generate question: {sentence}", max_length=50)
            question_text = generated_question[0]['generated_text']
            if question_text:  # Check if the generated question is not empty
                questions.append(question_text)

    return questions

# Step 4: Run the code with the path to your PDF file
pdf_path = "panduan aas.pdf"  # Change this to your PDF file path
text = extract_text_from_pdf(pdf_path)

if text:
    # Generate questions from the extracted text
    questions = generate_questions(text)
    
    # Display the generated questions
    for i, question in enumerate(questions):
        print(f"{i + 1}. {question}")
else:
    print("Could not process the PDF file.")
