from llm_router import LLMRouter

router = LLMRouter()

questions = [
    "Will frost occur tomorrow?",
    "What is the soil temperature?",
    "What action should I take to protect crops?"
]

for question in questions:

    agent = router.route(question)

    print("\nQuestion:")
    print(question)

    print("Selected Agent:")
    print(agent)