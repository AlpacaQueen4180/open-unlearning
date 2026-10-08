from pathlib import Path
import datetime,hashlib,json,math
root=Path('.').resolve();private=root/'work/spf-npo-gb200-20261006/private'
base=private/'checkpoint157-beta-completed-20261008'
manifest=json.loads((base/'manifest.private.json').read_bytes())
mapping={row['name']:row for row in manifest['files']}
sha=lambda raw:hashlib.sha256(raw).hexdigest()
def raw(name):
 row=mapping[name];value=(base/row['local_name']).read_bytes()
 assert len(value)==row['size'] and sha(value)==row['sha256']
 return value
def value(name):return json.loads(raw(name))
paired={}
for case in ['beta-09','beta-05']:
 summary=value(case+'/geometry-summary.private.json')
 rows=[json.loads(line) for line in raw(case+'/geometry.private.jsonl').splitlines()]
 assert len(rows)==32 and [row['step'] for row in rows]==list(range(1,33))
 assert summary['actual_updates']==32 and summary['actual_examples']==1024 and summary['warmup_steps']==125
 assert summary['geometry_sha256']==sha(raw(case+'/geometry.private.jsonl'))
 assert all(math.isfinite(v) for row in rows for v in row.values() if isinstance(v,float))
 nonzero=[r for r in rows if r['actual_update_norm']>0]
 assert len(nonzero)==31 and rows[0]['lr_before']==0 and rows[0]['actual_update_norm']==0
 assert summary['positive_actual_safety_dot_steps']==sum(r['actual_update_safety_dot']>0 for r in rows)
 assert summary['increased_anchor_loss_steps']==sum(r['anchor_loss_change']>0 for r in rows)
 paired[case]=dict(beta1=summary['beta1'],actual_updates=32,nonzero_updates=31,actual_examples=1024,
   positive_actual_update_safety_dot_steps=summary['positive_actual_safety_dot_steps'],
   increased_anchor_loss_steps=summary['increased_anchor_loss_steps'],
   negative_raw_gradient_dot_steps=sum(r['raw_task_safety_dot']<0 for r in rows),
   negative_projected_gradient_dot_steps=sum(r['projected_gradient_safety_dot']<0 for r in rows),
   positive_update_dot_despite_nonnegative_projected_gradient_dot_steps=sum(r['actual_update_safety_dot']>0 and r['projected_gradient_safety_dot']>=0 for r in rows),
   projected_gradient_dot_min=min(r['projected_gradient_safety_dot'] for r in rows),
   anchor_loss_at_start=rows[0]['anchor_loss_before'],anchor_loss_at_end=rows[-1]['anchor_loss_after'],
   observed_anchor_loss_change_sum=sum(r['anchor_loss_change'] for r in rows),
   actual_update_safety_cosine_min=min(r['actual_update_safety_cosine'] for r in nonzero),
   actual_update_safety_cosine_max=max(r['actual_update_safety_cosine'] for r in nonzero),
   geometry_sha256=summary['geometry_sha256'],summary_sha256=sha(raw(case+'/geometry-summary.private.json')),
   source_sha256=summary['source_sha256'],optimizer=summary['optimizer'])
tofu=value('checkpoint-157/evaluation/tofu/public-summary.json')
knowledge=value('checkpoint-157/evaluation/knowledge/public-summary.json')
ifbench=value('checkpoint-157/evaluation/ifbench-scores.private.json')
metrics={split:dict(count=item['count'],metrics={name:item['metrics'][name]['mean'] for name in ['normalized_exact_match','rouge_l_f1','mean_log_probability']}) for split,item in tofu['aggregates'].items()}
metrics.update(mmlu_aux=dict(count=knowledge['count'],correct=knowledge['correct'],accuracy=knowledge['accuracy'],num_fewshot=0),
 ifbench=dict(count=ifbench['count'],summary=ifbench['summary']))
assert knowledge['count']==1024 and ifbench['count']==300
e=dict(status='COMPLETED157_AND_BOUNDED_BETA_RESULTS_AGGREGATED',derived_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 archive_sha256=sha((base/'original.private.tar.gz').read_bytes()),immutable_export_file_count=62,
 source_sha256=sha(Path(__file__).read_bytes()),paired_beta=paired,checkpoint157_learning_capability=metrics,
 exact_row_outputs_kept_private=True,all32_steps_inside_original125_warmup=True,first_update_zero_learning_rate=True,
 per_step_dot_definition='actual parameter delta theta_after-theta_before dotted with original safety gradient',
 positive_dot_interpretation='positive first-order anchor-loss change; empirical post-step anchor loss recorded separately',
 harmfulness_reduction_verified=False,full_training_beta05_evaluated=False,causal_root_cause_verified=False,
 checkpoint157_harmfulness='NEW_JUDGE_PENDING',target_gate='insufficient_evidence',
 limitations=['32-step prefix only, original125-step warmup','same fixed anchor, no new anchors or task tuning',
 'scalar anchor-loss geometry does not replace safety benchmark labels','no repeated final checkpoint evaluations or16GB hashes'])
(private/'checkpoint157-beta-analysis-20261008.private.json').write_text(json.dumps(e,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(e))
