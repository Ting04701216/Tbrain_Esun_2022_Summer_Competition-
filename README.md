# README
## 程式架構
    ```
    $ tree -L 2

    .
    ├── api.py
    ├── data_preprocess
    │   ├── Phone2BoPoMo.py
    │   ├── prepare_aug_single.py
    │   └── prepare_input_single.py
    ├── inference
    │   ├── classify_news
    │   ├── inference.py
    │   ├── pinyin_translation
    │   └── spelling_correction
    ├── model
    │   ├── classify_news
    │   ├── pinyin_translation
    │   └── spelling_correction
    ├── README.md
    └── requirements.txt
    ```

## 程式環境
* 機器規格: CPU: i7-7700 3.60GHz, RAM: 128GB, GPU: Nvidia GTX1060 6G
* 系統平台: Ubuntu 16.04
* 程式語言: Python 3.7
* 函式庫: requirments.txt

## 執行範例

* 虛擬環境
    ```
    # 安裝虛擬環境
    $ python3 -m venv venv
    
    # 啟動虛擬環境 
    $ source venv/bin/activate
    
    # 安裝所需套件
    $ pip install -r requirements.txt 
    ```
* 資料前處理
    * 對文本資料做基本前處理，例如：'○' -> '零'，刪除標點符號等
    * 將發音資料轉換為拼音
    * 透過爬蟲新增資料(新聞源：今周刊)，將部分單詞替換成發音相似詞彙做為輸入資料，並轉換為拼音
    ```
    # 執行前處理
    # 原始資料
    $ python data_preprocess/prepare_input_single.py
    
    # 擴增資料(僅有文本)
    $ python data_preprocess/prepare_aug_single.py
    ```

* 模型訓練
    * 分為三個模型: 分類模型、翻譯模型以及糾錯模型
    * 分類模型
        - 針對是否為新聞語句，預先人工標註10000筆資料
        - 目標為分類資料為新聞語句或真實對話
        - 模型架構採用Bert
        - epoch=5, learning rate=2e-5, clip norm, warm up linear schedule
    * 翻譯模型
        - 將發音視為未知語言，目標為將其翻譯為正確文本
        - 模型架構採用transformer
        - epoch=20, learning rate=5e-4, dropout, clip norm, weight decay, label smoothing
        
    * 糾錯模型
        - 運用單詞發音與相似度等方法，將訓練資料與目標對齊之後，僅保留與目標語句結構長度一致的語句做為訓練樣本
        - 目標為偵測錯字，後續再利用mlm改錯([Integrated Semantic and Phonetic Post-correction for Chinese Speech Recognition](https://arxiv.org/abs/2111.08400))
        - 模型架構採用Bert
        - num_iter=160000, learning rate=2e-5
    
    ```
    # 執行模型訓練
    # 分類模型
    $ python classify_news/train.py
    
    # 翻譯模型
    $ python pinyin_translation/data.py
    $ bash pinyin_translation/data.sh
    $ bash pinyin_translation/training.sh
    
    # 糾錯模型
    $ python spelling_correction/data.py
    $ python spelling_correction/train_typo_detector.py --config spelling_correction/config_detect.py
    ```

* 模型推論
    * 由分類模型判別是否為新聞語句，再經翻譯與糾錯改錯模組產生多組預測結果，最後套用糾錯模型選取預測錯誤字數最少者做為結果輸出
    * API: gunicorn -w 2 --threads 6
    ```
    # 執行模型推論
    $ python inference.py
    ```