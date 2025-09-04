from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal, List, Dict, Any
import json

# Prompts

# prompt to be passed to the 2nd model in and 2nd step of llm call

class JsonSummarizer(BaseModel):
    summary: str = Field(description="A concise textual summary describing the overall content and key actions in the provided video segment, based on detected objects and their motion status.")
    events : str = Field(description="A single string listing all detected events across the video frames, including object names and their motion status, organized in chronological order.") 

output_parser_1 = PydanticOutputParser(pydantic_object=JsonSummarizer)

prompt1 = PromptTemplate(
    template="""You are a precise and objective video understanding assistant. You are given data in a dictionary where each key represents a specific timestamp in a video, formatted as "time_t" where t is the frame number.
For each timestamp key, the value is a list of multiple frames. This list always includes the current frame plus up to two frames before and two frames after it, giving a maximum of five frames per timestamp. At the start or end of the video, fewer surrounding frames will be included if they don't exist.
Each frame in the list is itself a list of detected objects. Every object is described as a dictionary containing a "caption" field, which is a short text description of the object, ending with either (steady) or (moving) to indicate whether the object is stationary or in motion.

Your tasks:
1. Use **only the provided captions** to describe what is happening.
2. Do **not add** any interpretation or assumptions.
3. Parse the motion status (inside parentheses) from each caption and consider it while summarizing.
4. Generate a detailed summary of the video based **only on the captions and motion status** across timestamps.
5. Identify and return key events with their respective timestamps.
6. Group similar actions or repeating events when possible.

### Input JSON Data:
```json
{video_json_here}
```
Provide the answer in the following JSON format:
{format_instructions}
""",
    input_variables=["video_json_here"],
    partial_variables={"format_instructions": output_parser_1.get_format_instructions()}
)





# prompt to be passed to the 1st model in and 2nd step of llm call if question asked about the video releated

class VideoAnswer(BaseModel):
    answer: str = Field(description='Answer the user question releated to summary provided.')
    source: str = 'video'

output_parser_2 = PydanticOutputParser(pydantic_object=VideoAnswer)

prompt2 = PromptTemplate(
    template="""You are a helpful assistant that understands video content. The user has uploaded a video, and the following summary was generated based on the detected events in the video:
            {VIDEO_SUMMARY}
            Now, the user asked the following question about the video:
            {USER_QUESTION}
            Using only the information from the video summary, answer the question in a clear and concise manner. If the information is not present in the summary, say "I can't find that information in the video."
            Provide the answer in the following JSON format:
            {format_instructions}
            """,
    input_variables=['VIDEO_SUMMARY', 'USER_QUESTION'],
    partial_variables={"format_instructions": output_parser_2.get_format_instructions()}
)



# prompt to be passed to the 1st model in and 2nd step of llm call if question asked general querry not about the video releated

class GeneralAnswer(BaseModel):
    answer: str = Field(description='Answer the user question based on your memory.')
    source: str = 'general'

output_parser_3 = PydanticOutputParser(pydantic_object=GeneralAnswer)

prompt3 = PromptTemplate(
    template="""You are a helpful general assistant. The user asked the following question:
            {USER_QUESTION}
            Answer concisely and clearly.
            {format_instructions}
            """,
    input_variables=['USER_QUESTION'],
    partial_variables={"format_instructions": output_parser_3.get_format_instructions()}        
)




# Prompt which wil execute first

class ToolSelectorOutput(BaseModel):
    tool_name: str = Field(description='Name the tool')

output_parser_4 = PydanticOutputParser(pydantic_object=ToolSelectorOutput)

prompt4 = PromptTemplate(
    template="""You are an intelligent assistant that decides which tool to use based on the user's question and context. There are two tools available:
1. **Tool Name**: `querry_ans`  
   **Description**: Use this tool when the user's question:
   - asks about details or events from the video (after a video has been uploaded and summarized),
   - or asks any general knowledge or reasoning question not related to video events or summary.

2. **Tool Name**: `video_event_summary`  
   **Description**: Use this tool **only if**:
   - the user specifically asks for a *summary* of the video or wants to know *what events occurred* in the video, AND
   - a new video has been uploaded (denoted by the variable `new = true`).
   - This tool must **not be called again** for the same video (`new = false`) as it would just return the same summary.

You will be given:
- the user's question: {USER_QUESTION}
- a boolean variable `new`: {NEW}

{format_instructions}

Respond only with a valid JSON object containing the tool name.
""",
    input_variables=["USER_QUESTION", "NEW"],
    partial_variables={"format_instructions": output_parser_4.get_format_instructions()}
)

dict = {}
dict['prompt1'] = prompt1.template
dict['prompt2'] = prompt2.template
dict['prompt3'] = prompt3.template
dict['prompt4'] = prompt4.template

# Dump to JSON file
with open("prompts.json", "w") as f:
    json.dump(dict, f, indent=4)


