model="gpt-4o-mini-2024-07-18"
# model="gpt-4o"
# model=llama370B
# model=qwen272B
# model=llama3

expdir=../exp/safety_${model}
mkdir -p $expdir

testfile=../data/CASEBench_data.json
mode=score
remove=source_accountability

python infer_api.py \
    --infile $testfile \
    --outfile $expdir/output_${mode}_${remove}.json \
    --model $model \
    --mode $mode \
    --remove $remove \

# Re-run twice to fill in the failed samples
python infer_api.py \
    --infile $expdir/output_${mode}_${remove}.json \
    --outfile $expdir/output_${mode}_${remove}.json \
    --model $model \
    --mode $mode \

python infer_api.py \
    --infile $expdir/output_${mode}_${remove}.json \
    --outfile $expdir/output_${mode}_${remove}.json \
    --model $model \
    --mode $mode \
