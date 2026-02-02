class InstructionsHandler:
    def __init__(self):
        self.asqp = {}

    def load_instruction_set1(self, count = 0):
        self.asqp['bos_instruct'] = """Definition: The output will be the quadruples consisting of (target, aspect, opinion, sentiment) in the input text and the sentiment polarity (pos, neg, other) of the opinion term. In cases where there are no quadruple the output should be notarget:none:none:none.
        Positive example 1-
        input: This phone is not very good , but compared to the iPhone , I think it is better than the iPhone except for the processor [ laughs cry ]
        output: iPhone:processor:better:pos
        Positive example 2-
        input: 778 Xiaomi Civi looks invincible and feels invincible [ doge ]
        output: Xiaomi Civi:looks:invincible:pos, Xiaomi Civi:feels:invincible:pos
        input: """
        self.asqp['eos_instruct'] = ' \noutput:'

    def load_instruction_set2(self, count=0):
        self.asqp['bos_instruct'] = """Definition: The output will be the quadruples consisting of (target, aspect, opinion, sentiment) in the input text and the sentiment polarity (pos, neg, other) of the opinion term. In cases where there are no quadruple the output should be notarget:none:none:none.
        Positive example 1-
        input: This phone is not very good , but compared to the iPhone , I think it is better than the iPhone except for the processor [ laughs cry ]
        output: iPhone:processor:better:pos
        Positive example 2-
        input: 778 Xiaomi Civi looks invincible and feels invincible [ doge ]
        output: Xiaomi Civi:looks:invincible:pos, Xiaomi Civi:feels:invincible:pos
        Negative example 1-
        input: Do n't tell me anything else , ca n't I afford a Huawei since I bought apple with more than 1W ? What does Huawei has except domestic ? The system is not good , it 's useless that a brand only has patriotic title .
        output: Huawei:system:not good:neg
        Negative example 2-
        input: Your apple has no high brush , no quick charge , low battery and perilously fragile , use more than 10,000 to buy such a mobile phone , what do you show off ? [ Puzzle ]
        output: apple:battery:low:neg
        Other example 1-
        input: How is the K40 photo compared with the realme GT NEO2 ?
        output: K40:photo:How:other
        Other example 2-
        input: Generally speaking , if you do n't care about wireless charging , 12X is definitely enough . After all , these two are only different from the processor and wireless charging .
        output: 12X:wireless charging:don't care about:other
        input: """
        self.asqp['eos_instruct'] = ' \noutput:'