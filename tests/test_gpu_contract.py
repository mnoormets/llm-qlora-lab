import pytest

def test_collator_masks_padding_even_when_pad_equals_eos():
    torch=pytest.importorskip('torch')
    from qlora.encoding import CompletionCollator
    rows=[{'input_ids':[4,9],'attention_mask':[1,1],'labels':[-100,9]},{'input_ids':[4,8,9],'attention_mask':[1,1,1],'labels':[-100,8,9]}]
    batch=CompletionCollator(9)(rows)
    assert batch['input_ids'].shape==(2,8)
    assert batch['labels'][0,1].item()==9 and (batch['labels'][0,2:]==-100).all()
    assert batch['attention_mask'][0,2:].sum().item()==0


def test_pinned_trainer_and_peft_interfaces_execute_on_cpu(tmp_path):
    torch=pytest.importorskip('torch');transformers=pytest.importorskip('transformers');peft=pytest.importorskip('peft')
    from qlora.encoding import CompletionCollator
    torch.set_num_threads(2)
    config=transformers.Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=64)
    model=transformers.Qwen2ForCausalLM(config)
    model=peft.get_peft_model(model,peft.LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','k_proj','v_proj','o_proj'],task_type='CAUSAL_LM'))
    rows=[{'input_ids':[1,2,3,4,5],'attention_mask':[1]*5,'labels':[-100,-100,3,4,5]}]*4
    args=transformers.TrainingArguments(output_dir=str(tmp_path),max_steps=1,per_device_train_batch_size=1,learning_rate=2e-4,report_to='none',save_strategy='no',use_cpu=True,remove_unused_columns=False)
    trainer=transformers.Trainer(model=model,args=args,train_dataset=rows,data_collator=CompletionCollator(0))
    result=trainer.train()
    import math
    assert math.isfinite(result.training_loss)
    assert all('lora_' in name for name,p in model.named_parameters() if p.requires_grad)


def test_t4_uses_native_bf16_check_not_emulation():
    from types import SimpleNamespace
    from qlora.train import preflight
    calls=[]
    def bf16(*,including_emulation):
        calls.append(including_emulation)
        return including_emulation
    cuda=SimpleNamespace(is_available=lambda:True,get_device_properties=lambda _:SimpleNamespace(name='Tesla T4',total_memory=15*1024**3),is_bf16_supported=bf16)
    assert preflight(SimpleNamespace(cuda=cuda))['bf16_supported'] is False
    assert calls==[False]


def test_child_failure_keeps_actual_traceback(tmp_path,capsys):
    import sys
    from qlora.runtime import run_logged
    log=tmp_path/'failed.log'
    with pytest.raises(RuntimeError,match='status 1'):
        run_logged([sys.executable,'-u','-c',"raise ValueError('actual training failure')"],log)
    assert 'ValueError: actual training failure' in log.read_text()
    assert 'ValueError: actual training failure' in capsys.readouterr().out


def test_logged_child_success(tmp_path,capsys):
    import sys
    from qlora.runtime import run_logged
    log=tmp_path/'success.log'
    assert run_logged([sys.executable,'-u','-c',"print('completed')"],log)==0
    assert log.read_text().strip()=='completed'
    assert 'completed' in capsys.readouterr().out
