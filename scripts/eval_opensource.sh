export CUDA_VISIBLE_DEVICES=0,1,2,3

# model=llama370B
model=dolphin
# model=qwen272B
# model=llama3
# model=mixtral8x7B

expdir=../exp/safety_${model}
mkdir -p $expdir

testfile=../data/CASEBench_data.json

mode=logits
remove=sender,recipient,transmission_principle


python infer_model.py \
    --infile $testfile \
    --outfile $expdir/output_${mode}.json \
    --model $model \
    --mode $mode \
    # --remove $remove \


python infer_model.py \
    --infile $expdir/output_${mode}.json \
    --outfile $expdir/output_${mode}.json \
    --model $model \
    --mode $mode \

python infer_model.py \
    --infile $expdir/output_${mode}.json \
    --outfile $expdir/output_${mode}.json \
    --model $model \
    --mode $mode \
