#=================参数设置=================
import multiprocessing as mp


class CFG:
    debug=False
    version = 'ex'    # 当前运行的版本号，会以此创建一个文件夹，注意命名区分！！！
    version_explanation = 'stage 2 reranker '

    #===== wandb =====
    wandb=False
    competition='Learning Equality - Curriculum Recommendations'

    #===== GPU Optimize Settingså =====
    apex=True
    gradient_checkpoint=False

    #===== State1 =====
    state1_model = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
    state1_path = './outputs/ex107/'
    state1_topk = 50
    
    #===== Train =====
    model= 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'

    epochs=4
    batch_size= 180
    max_len= 256
    encoder_lr = 3e-5 #2e-5 #[1e-5, 1e-5, 1e-5, 1e-5]
    decoder_lr = 3e-4
    layerwise_learning_rate_decay = 1.0
    seed=56   #XXX
    n_fold=4
    trn_fold = list(range(n_fold))
    num_workers = mp.cpu_count()
    print_each_epoch = 5 # 一个epoch打印的次数
    val_each_epoch = 2 # Main中自动计算,一个epoch中验证次数



    #===== tool ====
    scheduler='cosine'# ['linear', 'cosine']
    batch_scheduler=True
    num_cycles=0.5
    warmp_ratio = 0
    eps=1e-6
    betas=(0.9, 0.999)
    dropout=0.1
    weight_decay=0.01
    gradient_accumulation_steps=1
    max_grad_norm = 0.012


    #===== Path =====
    output_dir = f'./outputs/{version}/'  # {model}.split('/')[1]
    data_dir = './data/raw/'
    log_dir = './logs/'

