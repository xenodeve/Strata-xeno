"""Few-shot chain-of-thought prompts for the published-protocol arm. GSM8K: the first four exemplars of Wei et al. 2022."""

GSM8K_COT=(
    ('There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?',
     'There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.'),
    ('If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?',
     'There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.'),
    ('Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?',
     'Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.'),
    ('Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?',
     'Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.'),
)


def few_shot(question,shots):
    """The exemplars, then the question left open at `Answer:`."""
    if not 0<shots<=len(GSM8K_COT):raise ValueError('shots')
    return ''.join(f'Question: {q}\nAnswer: {a}\n\n' for q,a in GSM8K_COT[:shots])+f'Question: {question}\nAnswer:'
