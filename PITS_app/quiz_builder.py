from llama_index.core import load_index_from_storage, StorageContext
from llama_index.program.evaporate.df import DFRowsProgram
from llama_index.program.openai import OpenAIPydanticProgram
from global_settings import INDEX_STORAGE, QUIZ_SIZE, QUIZ_FILE
import pandas as pd

"""
First, we set up a DataFrame to structure the quiz questions and their associated options and
answers. This DataFrame will serve as the foundation for our quiz. It includes columns for the
question number, question text, four answer options, the correct answer, and a rationale for
the answer. The use of a pandas DataFrame will make handling and manipulating the quiz
data much easier.
"""
def build_quiz(topic):
    df = pd.DataFrame({
        "Question_no": pd.Series(dtype="int"),
        "Question_text": pd.Series(dtype="str"),
        "Option1": pd.Series(dtype="str"),
        "Option2": pd.Series(dtype="str"),
        "Option3": pd.Series(dtype="str"),
        "Option4": pd.Series(dtype="str"),
        "Correct_answer": pd.Series(dtype="str"),
        "Rationale": pd.Series(dtype="str"),
    })
    # Next, we need to load our vector index from storage. To do this, we must define a
    # StorageContext object while using the INDEX_STORAGE folder as a parameter:
    storage_context = StorageContext.from_defaults(
        persist_dir=INDEX_STORAGE
    )
    vector_index = load_index_from_storage(
    storage_context, index_id="vector")
    # Here, we used index_id to identify the vector index because there’s also a TreeIndex index
    # in that storage that we won’t be using for now. It’s time to initialize our DataFrame extractor:
    df_rows_program = DFRowsProgram.from_defaults(
        pydantic_program_cls=OpenAIPydanticProgram,
        df=df
    )
    # Now, we can define our query engine and craft a prompt that will generate the quiz questions:
    query_engine = vector_index.as_query_engine()
    quiz_query = (
        f"Create {QUIZ_SIZE} different quiz "
        "questions relevant for testing "
        "a candidate's knowledge about "
        f"{topic}. Each question will have 4 "
        "answer options. Questions must be "
        "general topic-related, not specific "
        "to the provided text. For each "
        "question, provide also the correct "
        "answer and the answer rationale. "
        "The rationale must not make any "
        "reference to the provided context, "
        "any exams or the topic name. Only "
        "one answer option should be correct."
    )
    response = query_engine.query(quiz_query)

    # Next, the prompt is passed to the query engine, and the response is then processed by
    # DFRowsProgram to convert it into a structured DataFrame format:
    result_obj = df_rows_program(input_str=response)
    new_df = result_obj.to_df(existing_df=df)
    new_df.to_csv(QUIZ_FILE, index=False)
    return new_df