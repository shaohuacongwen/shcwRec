# shcwRec

注意本地使用需要创建一些空文件夹

使用前需要用preprocess处理数据集文件

文件结构

```
$ tree
.
shcwRec/
    .gitignore
    env.txt
    执行方法.ipynb
    tsne.py
    run.sh
    data/
        book_data/
            book_item_cate.txt
            book_test.txt
            book_train.txt
            book_valid.txt
        rocket_data/
            rocket_cate_map.txt
            rocket_item_cate.txt
            rocket_item_map.txt
            rocket_test.txt
            rocket_train.txt
            rocket_user_map.txt
            rocket_valid.txt
    preprocess/
        category.py
        data.py
    src/
        data_iterator.py
        mostpop.py
        train.py
        model.py
        models/
            cx.py
            DNN.py
            GRU4Rec.py
            MIND.py
            ComiRec_SA.py
            ComiRec_DR.py
            UMI.py
    output/
    save_model/
        runs/
        best_model/
    results/
    ```
    
