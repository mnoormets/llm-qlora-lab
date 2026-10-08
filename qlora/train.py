"""Real 7B QLoRA experiment. Refuses CPU execution before downloading weights."""
import argparse,json,time,hashlib,platform
from pathlib import Path
from .data import ROOT,load_rows,score_predictions
from .encoding import encode_row,prompt_for,CompletionCollator
MODEL='Qwen/Qwen2.5-7B-Instruct'
REVISION='a09a35458c702b33eeacc393d103063234e8bc28'

def preflight(torch):
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required. No weights downloaded; no training result claimed.')
    properties=torch.cuda.get_device_properties(0)
    if properties.total_memory<14*1024**3:raise RuntimeError('At least 14 GiB GPU memory required for the initial 7B configuration')
    return {'name':properties.name,'memory_gib':properties.total_memory/1024**3,'bf16_supported':torch.cuda.is_bf16_supported(including_emulation=False)}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--steps',type=int,default=60);parser.add_argument('--eval-cases',type=int,default=16);parser.add_argument('--max-length',type=int,default=768);parser.add_argument('--preflight-only',action='store_true');args=parser.parse_args()
    if not 1<=args.steps<=1000 or not 1<=args.eval_cases<=40 or not 256<=args.max_length<=2048:parser.error('Invalid bounded experiment settings')
    import torch
    hardware=preflight(torch)
    if args.preflight_only:print(json.dumps(hardware,indent=2));return
    from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig,Trainer,TrainingArguments,set_seed
    from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
    set_seed(73);out=ROOT/'runs'/time.strftime('%Y%m%d-%H%M%S');out.mkdir(parents=True)
    dtype=torch.bfloat16 if hardware['bf16_supported'] else torch.float16
    report={'status':'running','model':MODEL,'revision':REVISION,'hardware':hardware,'dtype':str(dtype),'seed':73,'steps':args.steps,'max_length':args.max_length,'python':platform.python_version(),'torch':torch.__version__,'scope':'7B QLoRA on authored synthetic Estonian invoices; no real-contract accuracy claim','dataset_sha256':{name:hashlib.sha256((ROOT/'fixtures'/(name+'.jsonl')).read_bytes()).hexdigest() for name in ['train','validation','test']}}
    def save(): (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    save();print(json.dumps({'report':str(out/'report.json'),'hardware':hardware,'dtype':str(dtype)},indent=2),flush=True);started=time.perf_counter()
    try:
        tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,trust_remote_code=False)
        tokenizer.pad_token=tokenizer.eos_token;tokenizer.padding_side='right'
        quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=dtype)
        model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,trust_remote_code=False,use_safetensors=True,quantization_config=quant,device_map={'':0},dtype=dtype,attn_implementation='sdpa')
        train=[encode_row(tokenizer,row,args.max_length) for row in load_rows(ROOT/'fixtures/train.jsonl')]
        validation=[encode_row(tokenizer,row,args.max_length) for row in load_rows(ROOT/'fixtures/validation.jsonl')]
        test=load_rows(ROOT/'fixtures/test.jsonl')[:args.eval_cases]
        def evaluate_generation(candidate):
            candidate.eval();predictions=[];times=[]
            with torch.inference_mode():
                for row in test:
                    inputs=tokenizer(prompt_for(tokenizer,row),return_tensors='pt',add_special_tokens=False).to('cuda')
                    tick=time.perf_counter();generated=candidate.generate(**inputs,max_new_tokens=128,do_sample=False,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
                    torch.cuda.synchronize();times.append(time.perf_counter()-tick)
                    predictions.append(tokenizer.decode(generated[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True).strip())
            result=score_predictions(test,predictions);result['generation_seconds']=times;return result
        report['baseline']=evaluate_generation(model);save()
        model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True)
        model.config.use_cache=False
        model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,target_modules=['q_proj','k_proj','v_proj','o_proj'],bias='none',task_type='CAUSAL_LM'))
        trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
        if not trainable or any(p.requires_grad and 'lora_' not in name for name,p in model.named_parameters()):raise RuntimeError('Unexpected trainable base parameters')
        report['trainable_parameters']=trainable;report['lora']={'rank':16,'alpha':32,'target_modules':['q_proj','k_proj','v_proj','o_proj']};save()
        arguments=TrainingArguments(output_dir=str(out/'checkpoints'),max_steps=args.steps,per_device_train_batch_size=1,per_device_eval_batch_size=1,gradient_accumulation_steps=8,learning_rate=2e-4,warmup_ratio=.05,lr_scheduler_type='cosine',logging_steps=5,eval_strategy='steps',eval_steps=20,save_strategy='steps',save_steps=20,save_total_limit=2,bf16=hardware['bf16_supported'],fp16=not hardware['bf16_supported'],gradient_checkpointing=True,optim='paged_adamw_8bit',report_to='none',seed=73,data_seed=73,remove_unused_columns=False)
        trainer=Trainer(model=model,args=arguments,train_dataset=train,eval_dataset=validation,data_collator=CompletionCollator(tokenizer.pad_token_id))
        torch.cuda.reset_peak_memory_stats();result=trainer.train()
        model.save_pretrained(out/'adapter',safe_serialization=True);tokenizer.save_pretrained(out/'adapter')
        report['training_metrics']=result.metrics;report['training_history']=trainer.state.log_history
        report['peak_allocated_gpu_gib']=torch.cuda.max_memory_allocated()/1024**3;report['peak_reserved_gpu_gib']=torch.cuda.max_memory_reserved()/1024**3
        model.config.use_cache=True;report['adapted']=evaluate_generation(model)
        report['status']='completed';report['elapsed_seconds']=time.perf_counter()-started
        report['adapter_files']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/'adapter').iterdir() if p.is_file()};save()
        print(json.dumps({'report':str(out/'report.json'),'status':report['status'],'baseline':{k:v for k,v in report['baseline'].items() if k not in ['details','generation_seconds']},'adapted':{k:v for k,v in report['adapted'].items() if k not in ['details','generation_seconds']}},indent=2))
    except torch.cuda.OutOfMemoryError:
        report.update(status='failed_cuda_oom',elapsed_seconds=time.perf_counter()-started,recovery='Restart runtime; reduce max-length to 512. Keep batch size 1. Do not edit measured results.');save();raise
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__,error_message=str(error),elapsed_seconds=time.perf_counter()-started);save();raise
if __name__=='__main__':main()
