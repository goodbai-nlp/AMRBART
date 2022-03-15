def add_model_specific_args(parser, root_dir):
    parser.add_argument(
        "--train_data_file",
        default=None,
        type=str,
        required=True,
        help="The input training data file (a text file).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        help="The model architecture to be trained or fine-tuned.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="The data directory",
    )
    # Other parameters
    parser.add_argument(
        "--eval_data_file",
        default=None,
        type=str,
        help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
    )
    parser.add_argument(
        "--test_data_file",
        default=None,
        type=str,
        help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
    )

    parser.add_argument(
        "--config_name",
        default=None,
        type=str,
        help="Optional pretrained config name or path if not the same as model_name_or_path. If both are None, initialize a new config.",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        default=None,
        type=str,
        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path. If both are None, initialize a new tokenizer.",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)",
    )
    parser.add_argument(
        "--src_block_size",
        default=512,
        type=int,
        help="Optional input sequence length after tokenization.",
    )
    parser.add_argument(
        "--tgt_block_size",
        default=512,
        type=int,
        help="Optional input sequence length after tokenization.",
    )
    parser.add_argument(
        "--src_prefix",
        default="",
        type=str,
        help="Source prefix",
    )
    parser.add_argument(
        "--tgt_prefix",
        default="",
        type=str,
        help="Target prefix",
    )
    parser.add_argument(
        "--val_metric",
        default="bleu",
        type=str,
        help="validation metric",
        required=False,
        choices=["bleu", "rouge2", "loss", "smatch", None],
    )
    parser.add_argument(
        "--eval_beam",
        default=5,
        type=int,
        help="validation beams",
    )
    parser.add_argument(
        "--eval_lenpen",
        default=1.0,
        type=float,
        help="validation length penity",
    )
    parser.add_argument(
        "--eval_max_length",
        default=512,
        type=int,
        help="Max tgt generated length",
    )
    parser.add_argument(
        "--label_smoothing",
        default=0.0,
        type=float,
        help="Label smoothed Cross Entorpy Loss",
    )
    parser.add_argument(
        "--dropout",
        default=None,
        type=float,
        help="Dropout for model",
    )
    parser.add_argument("--unified_input", action="store_true", help="Whether to use unified input for finetuning.")
    parser.add_argument("--smart_init", action="store_true", help="Whether to initialize AMR word embeddings smartly.")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument(
        "--do_predict", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Run evaluation during training at each logging step.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=4,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=4,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--train_num_workers",
        default=4,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--eval_num_workers",
        default=4,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--process_num_workers",
        default=1,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=0,
        help="Number of updates steps to control early stop",
    )
    parser.add_argument(
        "--lr_scheduler", default="linear", type=str, help="The initial learning rate for Adam."
    )
    parser.add_argument(
        "--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam."
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps."
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite the content of the output directory",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O2",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=1,
        help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=-1,
        help="save step interval",
    )
    parser.add_argument("--resume", action="store_true", help="Whether to continue run training.")
    return parser