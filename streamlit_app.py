# aptitude_quiz_app.py

import streamlit as st
import google.generativeai as genai
import time
import re
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple

# --- Configuration & Constants ---
NUM_QUESTIONS_PER_QUIZ = 3  # Adjust as needed
RETRY_DELAY_SECONDS = 5 # For rate limiting
MAX_RETRIES = 3

# --- Pydantic Models ---
class QuizQuestion(BaseModel):
    id: int
    text: str
    role_category: str
    # We could add 'expected_answer_keywords' or similar if we wanted more structured evaluation
    # but for now, we'll let Gemini handle open-ended evaluation.

class EvaluationResult(BaseModel):
    assessment: str  # e.g., "Correct", "Partially Correct", "Incorrect"
    feedback: str
    points: int

# --- Gemini API Functions ---

@st.cache_resource(ttl=3600) # Cache the client for 1 hour
def get_gemini_client(api_key: str):
    """Initializes and returns the Gemini Pro client."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name='gemini-1.5-flash', # or 'gemini-pro' - flash is faster and cheaper
            generation_config={"temperature": 0.7}, # Slight creativity
             # Adjust safety settings if needed, be mindful of implications
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ]
        )
        return model
    except Exception as e:
        st.error(f"Error initializing Gemini client: {e}")
        return None

def make_gemini_call(model, prompt_text: str, is_json_output: bool = False):
    """Makes a call to Gemini model with retries for rate limiting."""
    for attempt in range(MAX_RETRIES):
        try:
            # For potential JSON mode if Gemini API supports it directly in the future
            # response_mime_type = "application/json" if is_json_output else "text/plain"
            # response = model.generate_content(prompt_text, generation_config=genai.types.GenerationConfig(response_mime_type=response_mime_type))

            response = model.generate_content(prompt_text)
            
            # Check for empty or blocked responses
            if not response.parts:
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                    st.warning(f"Gemini API call blocked. Reason: {response.prompt_feedback.block_reason}")
                    return f"Error: Content blocked due to {response.prompt_feedback.block_reason}"
                else:
                    st.warning("Gemini API call returned an empty response.")
                    return "Error: Empty response from API."
            return response.text
        except (genai.types.generation_types.StopCandidateException) as e:
            st.warning(f"Gemini API call stopped. This might be due to safety settings or other reasons. {e}")
            return f"Error: Content generation stopped. {e}"
        except (
            genai.types.generation_types.BlockedPromptException, 
            genai.types.generation_types.InvalidArgumentException
        ) as e: # Specific exceptions related to prompt issues
             st.error(f"Error with prompt or generation: {e}")
             return f"Error: {e}" # No retry for these
        except Exception as e: # Catch broader exceptions, including potential rate limits
            # This is a generic catch. The google-generativeai library might have more specific
            # exceptions for rate limits like google.api_core.exceptions.ResourceExhausted
            # but it can vary. Check the library's documentation for the exact exception.
            if "rate limit" in str(e).lower() or "resource has been exhausted" in str(e).lower() or "429" in str(e).lower() :
                st.warning(f"Rate limit likely hit. Retrying in {RETRY_DELAY_SECONDS}s (Attempt {attempt+1}/{MAX_RETRIES})")
                time.sleep(RETRY_DELAY_SECONDS)
                if attempt + 1 == MAX_RETRIES:
                    st.error(f"Max retries reached for Gemini API call. Error: {e}")
                    return f"Error: Max retries for API call. {e}"
            else:
                st.error(f"An unexpected error occurred during Gemini API call: {e}")
                return f"Error: {e}" # No retry for unknown errors
    return "Error: Failed to get response from Gemini after multiple retries."


def generate_quiz_questions(model, age: int, position: str, num_questions: int) -> List[QuizQuestion]:
    """Generates quiz questions using Gemini."""
    questions = []
    base_prompt = (
        f"You are an aptitude test question generator. Create {num_questions} aptitude questions "
        f"suitable for a candidate aged {age} applying for the role of '{position}'. "
        f"The questions should be answerable with text, explanations, or pseudo-code, not requiring actual code compilation. "
        f"Focus on problem-solving, logical reasoning, or core concepts relevant to the role category. "
        f"Each question should be distinct.\n\n"
        f"For each question, output it in the format:\n"
        f"Question <N>: <The question text>\n\n"
        f"Example Role Categories and Question types:\n"
        f"- IoT: Conceptual question about sensor data processing challenges.\n"
        f"- Blockchain: Question about the difference between PoW and PoS.\n"
        f"- Cybersecurity: Scenario-based question about identifying a phishing attempt.\n"
        f"- DevOps: Question about the benefits of CI/CD.\n"
        f"- AI/ML: Conceptual question about overfitting.\n"
        f"- Software Developer (Python/Java/etc.): Pseudo-code for a simple algorithm like reversing a string or finding duplicates.\n\n"
        f"Generate the questions now for role: {position}."
    )
    
    response_text = make_gemini_call(model, base_prompt)

    if response_text.startswith("Error:"):
        st.error(f"Failed to generate questions: {response_text}")
        return []

    # Parsing the questions
    # Expecting format "Question <N>: <Text>"
    raw_questions = re.findall(r"Question \d+:\s*(.+)", response_text, re.IGNORECASE)

    if not raw_questions:
        # Fallback if primary parsing fails
        raw_questions = [q.strip() for q in response_text.split('\n\n') if q.strip() and not q.lower().startswith("question ")]
        raw_questions = [q.split(":", 1)[1].strip() if ":" in q else q for q in raw_questions if q.strip()]

    for i, q_text in enumerate(raw_questions):
        if q_text:
            questions.append(QuizQuestion(id=i, text=q_text.strip(), role_category=position))
        if len(questions) >= num_questions:
            break
    
    if not questions:
        st.warning("Could not parse any questions from Gemini's response. The response was:")
        st.text_area("Gemini Response", response_text, height=200)
        # Add a generic fallback if generation truly fails
        questions.append(QuizQuestion(id=0, text=f"Describe a challenging project you worked on related to {position} and how you overcame obstacles.", role_category=position))

    return questions[:num_questions]


def evaluate_user_answer(model, question: QuizQuestion, user_answer: str, age: int) -> EvaluationResult:
    """Evaluates the user's answer using Gemini and attempts to parse a structured response."""
    prompt = (
        f"Role: Aptitude Test Evaluator\n"
        f"You are evaluating an answer to an aptitude question for a candidate aged {age} "
        f"applying for a role in '{question.role_category}'.\n\n"
        f"Question: {question.text}\n\n"
        f"Candidate's Answer: {user_answer}\n\n"
        f"Please evaluate the answer. Consider correctness, clarity, and relevance to the question and role. "
        f"Provide your assessment in the following strict format:\n"
        f"Assessment: [Correct/Partially Correct/Incorrect]\n"
        f"Feedback: [Your detailed feedback for the candidate]\n"
        f"Points: [Allocate points: 2 for Correct, 1 for Partially Correct, 0 for Incorrect]"
    )
    
    response_text = make_gemini_call(model, prompt)

    if response_text.startswith("Error:"):
        return EvaluationResult(assessment="Error", feedback=response_text, points=0)

    try:
        assessment_match = re.search(r"Assessment:\s*(Correct|Partially Correct|Incorrect)", response_text, re.IGNORECASE)
        feedback_match = re.search(r"Feedback:\s*(.+)", response_text, re.DOTALL | re.IGNORECASE) # Check if DOTALL affects points
        points_match = re.search(r"Points:\s*(\d+)", response_text, re.IGNORECASE)

        assessment = "Undetermined"
        feedback = "Could not parse feedback from Gemini."
        points = 0

        if assessment_match:
            assessment_str = assessment_match.group(1).strip()
            if "partially correct" in assessment_str.lower(): assessment = "Partially Correct"
            elif "correct" in assessment_str.lower(): assessment = "Correct"
            elif "incorrect" in assessment_str.lower(): assessment = "Incorrect"
        
        if feedback_match:
            # Refine feedback parsing to stop before "Points:" if it exists and DOTALL grabbed too much
            raw_feedback = feedback_match.group(1).strip()
            if points_match and "Points:" in raw_feedback:
                feedback = raw_feedback.split("Points:")[0].strip()
            else:
                feedback = raw_feedback

        if points_match:
            points = int(points_match.group(1).strip())
        elif assessment != "Undetermined": # Fallback points based on assessment string
            if assessment == "Correct": points = 2
            elif assessment == "Partially Correct": points = 1
            else: points = 0
        
        if assessment == "Undetermined" and not feedback_match: # If all parsing fails
             return EvaluationResult(assessment="Error", feedback=f"Could not parse Gemini evaluation. Response: {response_text}", points=0)


        return EvaluationResult(assessment=assessment, feedback=feedback, points=points)

    except Exception as e:
        st.error(f"Error parsing evaluation: {e}. Raw response:\n{response_text}")
        return EvaluationResult(assessment="Error", feedback=f"Parsing error: {e}", points=0)

# --- Streamlit App UI ---

st.set_page_config(layout="wide", page_title="CS Aptitude Quiz Chat")
st.title("📝 CS Aptitude Quiz Chat")
st.markdown("Get aptitude questions based on your age and target CS role. Type your answers and get them evaluated by AI!")

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("Quiz Setup")
    api_key = st.text_input("Enter your Google Gemini API Key:", type="password")

    if not api_key:
        st.warning("Please enter your Gemini API key to use the app.")
        st.stop()

    # Attempt to load model, store in session_state to avoid re-init on every script run
    if 'gemini_model' not in st.session_state or st.session_state.gemini_model is None:
        with st.spinner("Initializing AI Model..."):
            st.session_state.gemini_model = get_gemini_client(api_key)
    
    if st.session_state.gemini_model is None:
        st.error("Failed to initialize Gemini Model. Check API key and console for errors.")
        st.stop()
    
    model = st.session_state.gemini_model

    # Initialize session state variables
    if "quiz_started" not in st.session_state:
        st.session_state.quiz_started = False
    if "questions" not in st.session_state:
        st.session_state.questions = []
    if "current_q_index" not in st.session_state:
        st.session_state.current_q_index = 0
    if "user_score" not in st.session_state:
        st.session_state.user_score = 0
    if "total_possible_score" not in st.session_state:
        st.session_state.total_possible_score = 0
    if "chat_history" not in st.session_state: # To store Q&A flow
        st.session_state.chat_history = []
    if "answer_submitted_for_current_q" not in st.session_state:
        st.session_state.answer_submitted_for_current_q = False

    # --- User Inputs ---
    age = st.number_input("Your Age:", min_value=16, max_value=70, value=25, step=1,
                          disabled=st.session_state.quiz_started)
    
    cs_roles = [
        "AI/ML Engineer", "Data Scientist", "Machine Learning Researcher",
        "IoT Solutions Architect", "Embedded Systems Developer (IoT)",
        "Blockchain Developer", "Smart Contract Developer",
        "Cybersecurity Analyst", "Penetration Tester", "Security Engineer",
        "DevOps Engineer", "Cloud Engineer (AWS/Azure/GCP)", "Site Reliability Engineer",
        "Software Developer (Python)", "Software Developer (Java)", "Software Developer (JavaScript/TypeScript)",
        "Frontend Developer", "Backend Developer", "Full-Stack Developer",
        "Game Developer", "Mobile App Developer (Android/iOS)",
        "Database Administrator", "Network Engineer"
    ]
    position = st.selectbox("Applying Position:", options=cs_roles, index=0,
                            disabled=st.session_state.quiz_started)

    # --- Control Buttons ---
    if not st.session_state.quiz_started:
        if st.button("🚀 Start Quiz", type="primary", use_container_width=True):
            with st.spinner(f"Generating {NUM_QUESTIONS_PER_QUIZ} questions for {position}... This may take a moment."):
                generated_qs = generate_quiz_questions(model, age, position, NUM_QUESTIONS_PER_QUIZ)
            if generated_qs:
                st.session_state.questions = generated_qs
                st.session_state.quiz_started = True
                st.session_state.current_q_index = 0
                st.session_state.user_score = 0
                st.session_state.total_possible_score = len(generated_qs) * 2 # Max 2 points per question
                st.session_state.chat_history = []
                st.session_state.answer_submitted_for_current_q = False
                st.success(f"Quiz started with {len(st.session_state.questions)} questions!")
                st.rerun() # Rerun to update the main page display
            else:
                st.error("Could not generate questions. Please try again or check the role.")
    else: # Quiz is in progress
        if st.button("🔁 Restart Quiz", use_container_width=True):
            # Reset all relevant session state variables
            for key in ['quiz_started', 'questions', 'current_q_index', 'user_score',
                        'total_possible_score', 'chat_history', 'answer_submitted_for_current_q', 'user_age', 'user_position']:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.gemini_model = None # Force re-init of model if API key changes.
            st.rerun()

# --- Main Quiz Area ---
if not st.session_state.quiz_started:
    st.info("Adjust the settings in the sidebar and click 'Start Quiz'.")

else: # Quiz is active
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Check if quiz is over
    if st.session_state.current_q_index >= len(st.session_state.questions):
        st.balloons()
        st.success(f"🎉 Quiz Finished! 🎉")
        st.subheader(f"Your Final Score: {st.session_state.user_score} / {st.session_state.total_possible_score}")
        
        final_message = f"**Quiz Summary for {st.session_state.get('user_position', position)} (Age: {st.session_state.get('user_age', age)})**\n\n"
        final_message += f"Final Score: {st.session_state.user_score} out of {st.session_state.total_possible_score}.\n\n"
        final_message += "Here's a recap of your performance:\n"
        for item in st.session_state.chat_history:
            if item['role'] == 'user':
                final_message += f"\n**Your Answer:**\n{item['content']}\n"
            elif item['role'] == 'assistant' and "Question:" not in item['content']: # Filter out original question display
                 final_message += f"\n**AI Feedback:**\n{item['content']}\n"
        
        st.download_button(
            label="📥 Download Quiz Summary",
            data=final_message,
            file_name=f"quiz_summary_{position.replace(' ', '_')}.txt",
            mime="text/plain"
        )
        st.info("Click 'Restart Quiz' in the sidebar to try again with new questions or settings.")

    else: # Quiz in progress, show current question
        current_question: QuizQuestion = st.session_state.questions[st.session_state.current_q_index]

        # Display current question (only once per question)
        if not st.session_state.chat_history or \
           (st.session_state.chat_history and not st.session_state.chat_history[-1]["content"].startswith(f"**Question {st.session_state.current_q_index + 1}")):
            
            question_display = f"**Question {st.session_state.current_q_index + 1} of {len(st.session_state.questions)} (Role: {current_question.role_category})**\n\n{current_question.text}"
            st.session_state.chat_history.append({"role": "assistant", "content": question_display})
            with st.chat_message("assistant"):
                st.markdown(question_display)
            st.session_state.answer_submitted_for_current_q = False # Reset for new question

        # Get user's answer via chat_input
        user_typed_answer = st.chat_input("Your answer (type text or pseudo-code)...", key=f"answer_q_{st.session_state.current_q_index}")

        if user_typed_answer and not st.session_state.answer_submitted_for_current_q:
            st.session_state.user_age = age # Store for potential use if quiz restarts without new sidebar input
            st.session_state.user_position = position
            
            # Add user answer to chat
            st.session_state.chat_history.append({"role": "user", "content": user_typed_answer})
            with st.chat_message("user"):
                st.markdown(user_typed_answer)

            # Evaluate answer
            with st.spinner("AI is evaluating your answer..."):
                evaluation: EvaluationResult = evaluate_user_answer(model, current_question, user_typed_answer, age)

            # Update score and display feedback
            st.session_state.user_score += evaluation.points
            
            feedback_message = (f"**Assessment:** {evaluation.assessment} ({evaluation.points} points)\n\n"
                                f"**Feedback:**\n{evaluation.feedback}\n\n"
                                f"*Current Total Score: {st.session_state.user_score} / {st.session_state.total_possible_score}*")
            
            st.session_state.chat_history.append({"role": "assistant", "content": feedback_message})
            with st.chat_message("assistant"):
                st.markdown(feedback_message)
            
            st.session_state.answer_submitted_for_current_q = True # Mark as submitted

        # "Next Question" button logic (or automatically proceed if preferred)
        if st.session_state.answer_submitted_for_current_q:
            if st.button("➡️ Next Question", key=f"next_q_btn_{st.session_state.current_q_index}", type="primary"):
                st.session_state.current_q_index += 1
                st.session_state.answer_submitted_for_current_q = True
                #st.session_state.answer_submitted_for_current_q = False # Reset for the next question
                #st.rerun()
