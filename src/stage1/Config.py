#=================参数设置=================
import multiprocessing as mp
import os

class CFG:
    debug=False
    version = ''    # 当前运行的版本号，会以此创建一个文件夹，注意命名区分！！！
    version_explanation = 'stage 1 retrieval '

    #===== wandb =====
    wandb=False
    competition='Learning Equality - Curriculum Recommendations'

    #===== GPU Optimize Settingså =====
    apex=True
    gradient_checkpoint=True

    #===== Train =====
    model = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
    n_fold=4
    trn_fold = list(range(n_fold))
    batch_size = 360
    max_len = 128
    epochs=15
    print_each_epoch = 4 # 一个epoch打印的次数
    val_each_epoch = 2   # Main中自动计算,一个epoch中验证次数
    top_k = [5, 20, 50]  
    metric_to_track = "recall@50"   
 

    encoder_lr = 5e-5 
    decoder_lr = 5e-4
    layerwise_learning_rate_decay = 1.0
    seed=56   #XXX
    num_workers = mp.cpu_count()


    #===== tool ====
    scheduler='cosine'# ['linear', 'cosine']
    batch_scheduler=True
    num_cycles=0.5
    warmp_ratio = 0
    eps=1e-6
    betas=(0.9, 0.999)
    dropout=0.1
    weight_decay=0.001
    gradient_accumulation_steps=1
    

    #===== Path =====
    output_dir = f'./outputs/{version}/' 
    data_dir = './data/raw/'
    log_dir = './logs/'    
    stage1_data_dir = './data/stage1/'
