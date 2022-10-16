#:<<!
gold=$1
gold_tok=$1.tok
pred=$2
pred_tok=$2.tok

tokenizer=cdec-corpus/tokenize-anything.sh
echo "Tokenizing files ..."
bash $tokenizer -u  < $gold > $gold_tok
bash $tokenizer -u  < $pred > $pred_tok
#!

echo "Evaluating BLEU score ..."
python eval_gen.py --in-tokens $pred_tok --in-reference-tokens $gold_tok

echo "Evaluating Meteor score ..."
java -jar meteor-1.5/meteor-1.5.jar $pred_tok $gold_tok > $pred.meteor
tail -n 10 $pred.meteor

