from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI

import os
os.environ['open_ai'] 

llm = ChatOpenAI(model='gpt-4o-mini')
# Agents
explainer_agent = Agent(
    role="Plant Pathologist",
    goal="Explain the disease in a clear, short way for farmers",
    backstory="Expert in identifying and describing rice plant diseases.",
    llm=llm
)

treatment_agent = Agent(
    role="Agronomist",
    goal="Suggest best treatment for the disease in rice",
    backstory="Knows organic and chemical treatment strategies.",
    llm=llm
)

risk_agent = Agent(
    role="Field Advisor",
    goal="Assess risk level of disease on farmer’s crop",
    backstory="Helps farmers avoid crop loss by early intervention.",
    llm=llm
)

# Task definitions
def get_diagnosis_agents_pipeline(disease_label):
    disease = disease_label.capitalize()

    task1 = Task(
        description=f"A rice plant is infected with **{disease}**. Explain the disease in detail.",
        expected_output="Concise explanation of the disease for farmers.",
        agent=explainer_agent
    )

    task2 = Task(
        description=f"Suggest effective treatment for **{disease}** in rice crops.",
        expected_output="Short treatment plan with any precautions.",
        agent=treatment_agent
    )

    task3 = Task(
        description=f"How risky is **{disease}** for a farmer's crop? Keep it short.",
        expected_output="Risk level and urgency of action.",
        agent=risk_agent
    )

    crew = Crew(
        agents=[explainer_agent, treatment_agent, risk_agent],
        tasks=[task1, task2, task3],
        verbose=True
    )

    result = crew.kickoff()
    return result