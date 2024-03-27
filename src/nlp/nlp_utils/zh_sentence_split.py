from sentence_spliter.logic_graph import (
    long_short_cuter,
    simple_cuter,
)  # V1.2.4; 依赖包 attrdict 过时，它的import需修改 collections -> collections.abc
from sentence_spliter.automata.state_machine import StateMachine
from sentence_spliter.automata.sequence import StrSequence


def sent_tokenize_zh(intext, simple_cut):
    if not simple_cut:
        cuter = StateMachine(
            #  long_short_cuter()  # not so good
            long_short_cuter(hard_max=16, max_len=64, min_len=8)
        )
    else:  # use simple_cuter
        cuter = StateMachine(simple_cuter())
    sequence = cuter.run(StrSequence(intext))
    sentences = sequence.sentence_list()
    return sentences
