import json
import pytest
from qlora.data import make_row,Extraction,parse_output,score_predictions
from qlora.encoding import encode_row

def test_disjoint_entities_and_template_sets():
    sets=[{make_row(i,t)['id'] for i in ids} for ids,t in [(range(400),0),(range(400,440),3),(range(440,480),4)]]
    assert not sets[0]&sets[1] and not sets[1]&sets[2] and not sets[0]&sets[2]
    for index in range(480):Extraction.model_validate(make_row(index,index%6)['expected'])

@pytest.mark.parametrize('text',['{}','{"reference":null,"amount":1,"currency":"EUR","due_date":null}','{"reference":null,"amount":"1.1","currency":"EUR","due_date":null}','{"reference":null,"amount":null,"currency":null,"due_date":"2027-02-30"}','{"reference":null,"reference":"x","amount":null,"currency":null,"due_date":null}'])
def test_invalid_or_duplicate_json_is_rejected(text):
    with pytest.raises(ValueError):parse_output(text)

def test_scoring_keeps_failures_and_missing_fields():
    rows=[make_row(1,0),make_row(2,1)]
    result=score_predictions(rows,[json.dumps(rows[0]['expected']),'not json'])
    assert result['exact_match_fraction']==.5 and result['valid_json_fraction']==.5 and result['field_accuracy']==.5
    assert result['details'][1]['parsed'] is None

class Tokenizer:
    eos_token='!'
    def encode(self,text,**kwargs):return [ord(c) for c in text]
    def apply_chat_template(self,messages,**kwargs):return 'prompt:'

def test_prompt_mask_does_not_train_on_input_and_retains_eos():
    row=make_row(4,0);encoded=encode_row(Tokenizer(),row)
    assert encoded['labels'][:7]==[-100]*7
    assert encoded['labels'][7:]==encoded['input_ids'][7:]
    assert encoded['labels'][-1]==ord('!')
    with pytest.raises(ValueError):encode_row(Tokenizer(),row,max_length=10)

def test_preflight_stops_before_model_download():
    from qlora.train import preflight
    class Cuda:
        def is_available(self):return False
    class Torch:cuda=Cuda()
    with pytest.raises(RuntimeError,match='No weights downloaded'):preflight(Torch())
