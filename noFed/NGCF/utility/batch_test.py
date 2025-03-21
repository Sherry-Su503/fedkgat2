# This file is based on the NGCF author's implementation
# <https://github.com/xiangwang1223/neural_graph_collaborative_filtering/blob/master/NGCF/utility/batch_test.py>.
# It implements the batch test.

import heapq
import multiprocessing
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

import utility.metrics as metrics
from utility.load_data import *
from utility.parser import parse_args

cores = multiprocessing.cpu_count()

args = parse_args()
Ks = eval(args.Ks)


data_generator = Data(
    args.data_path, args.dataset, batch_size=args.batch_size
)
USR_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_items
N_TRAIN, N_TEST = data_generator.n_train, data_generator.n_test
BATCH_SIZE = args.batch_size


def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)#预测评分最大的k个item

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc,f1 = get_auc(item_score, user_pos_test)
    # print(item_score,'----',user_pos_test)
    # auc = roc_auc_score(item_score, user_pos_test)
    return r, auc,f1


def get_auc(item_score, user_pos_test):
#     # item_score = sorted(item_score.items(), key=lambda kv: kv[1])
#     # item_score.reverse()
#     # item_sort = [x[0] for x in item_score]
#     # posterior = [x[1] for x in item_score]
#     item_sort, posterior = zip(*sorted(item_score.items(), key=lambda kv: kv[1], reverse=True))
#     # 生成真实标签：如果 item 在 user_pos_test 中，则为 1，否则为 0
#     y_true = [1 if i in user_pos_test else 0 for i in item_sort]

#     # 计算 AUC
#     try:
#         auc = roc_auc_score(y_true, posterior)  # 真实 0/1 标签 vs. 预测分数
#     except ValueError:
#         auc = 0.0  # 若 AUC 计算失败，默认设为 0

#     # 计算 F1
#     threshold = 0  # 设定二分类阈值
#     y_pred = [1 if score >= threshold else 0 for score in posterior]  # 生成二分类预测
#     f1 = f1_score(y_true, y_pred)  # 计算 F1
#     return auc,f1
      # 正确计算AUC和F1
    y_true = [1 if item in user_pos_test else 0 for item in item_score.keys()]
    y_score = list(item_score.values())  # 原始分数

    # 计算AUC
    auc = roc_auc_score(y_true, y_score)

    # 计算F1（假设二分类概率输出）
    threshold = np.median(y_score) 
    # print('y_score',y_score)
    # breakpoint()
    y_pred = [1 if score >= threshold else 0 for score in y_score]
    f1 = f1_score(y_true, y_pred, average='binary')
    return auc,f1


def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc,f1 = get_auc(item_score, user_pos_test)
    # print(item_score,'----',user_pos_test)
    # auc = roc_auc_score(item_score, user_pos_test)
    return r, auc,f1


def get_performance(user_pos_test, r, auc, f1,Ks):
    precision, recall, ndcg, hit_ratio = [], [], [], []

    for K in Ks:
        pre = metrics.precision_at_k(r, K)
        rec = metrics.recall_at_k(r, K, len(user_pos_test))
        precision.append(pre)
        recall.append(rec)
        ndcg.append(metrics.ndcg_at_k(r, K))
        hit_ratio.append(metrics.hit_at_k(r, K))
        # f1_scores.append(metrics.F1(pre, rec))
    
    # # print('user_pos_test',user_pos_test)
    # target = [1 if i in user_pos_test else 0 for i in range(len(posterior))]
    # threshold = 0.5  # 设置分类阈值，可调整
    # output_binary = (np.array(posterior) >= threshold).astype(int)
    # # output_binary = [1 if score >= threshold else 0 for score in posterior]
    # f1_global = f1_score(target, output_binary)  # 全局 F1 计算

    return {
        "recall": np.array(recall),
        "precision": np.array(precision),
        "ndcg": np.array(ndcg),
        "hit_ratio": np.array(hit_ratio),
        "auc": auc,
        "f1": f1,
    }


def test_one_user(x):
    # user u's ratings for user u
    rating = x[0]
    # uid
    u = x[1]
    # user u's items in the training set
    try:
        training_items = data_generator.train_items[u]
    except Exception:
        training_items = []
    # user u's items in the test set
    user_pos_test = data_generator.test_set[u]

    all_items = set(range(ITEM_NUM))

    test_items = list(all_items - set(training_items))

    if args.test_flag == "part":
        r, auc,f1 = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)
    else:
        r, auc,f1 = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)
     # 获取所有评分，作为 posterior 传递给 F1 计算
    # print('11')
    # posterior = [rating[i] for i in test_items]
    # item_sort, posterior = zip(*sorted(item_score.items(), key=lambda kv: kv[1], reverse=True))
    # print('222')

    return get_performance(user_pos_test, r, auc,f1, Ks)


def test(model, g, users_to_test, batch_test_flag=False):
    result = {
        "precision": np.zeros(len(Ks)),
        "recall": np.zeros(len(Ks)),
        "ndcg": np.zeros(len(Ks)),
        "hit_ratio": np.zeros(len(Ks)),
        "auc": 0.0,
        "f1": 0.0,
    }

    # pool = multiprocessing.Pool(cores)

    u_batch_size = 10000
    i_batch_size = BATCH_SIZE

    test_users = users_to_test
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0
    f1_list = []  # 存储每个用户的 F1

    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start:end]

        if batch_test_flag:
            # batch-item test
            n_item_batchs = ITEM_NUM // i_batch_size + 1
            rate_batch = np.zeros(shape=(len(user_batch), ITEM_NUM))
            # test_items = list(set(range(ITEM_NUM)) - set(training_items))
            # rate_batch = np.zeros((len(user_batch), len(test_items)))

            i_count = 0
            for i_batch_id in range(n_item_batchs):
                i_start = i_batch_id * i_batch_size
                i_end = min((i_batch_id + 1) * i_batch_size, ITEM_NUM)

                item_batch = range(i_start, i_end)

                u_g_embeddings, pos_i_g_embeddings, _ = model(
                    g, "user", "item", user_batch, item_batch, []
                )
                i_rate_batch = (
                    model.rating(u_g_embeddings, pos_i_g_embeddings)
                    .detach()
                    .cpu()
                )

                rate_batch[:, i_start:i_end] = i_rate_batch
                i_count += i_rate_batch.shape[1]

            assert i_count == ITEM_NUM

        else:
            # all-item test
            item_batch = range(ITEM_NUM)
            u_g_embeddings, pos_i_g_embeddings, _ = model(
                g, "user", "item", user_batch, item_batch, []
            )
            rate_batch = (
                model.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()
            )

        user_batch_rating_uid = zip(rate_batch.numpy(), user_batch)
        # user_batch_rating_uid = zip(rate_batch, user_batch)
        with multiprocessing.Pool(processes=cores) as pool:
            batch_result = pool.map(test_one_user, user_batch_rating_uid)

        # batch_result = pool.map(test_one_user, user_batch_rating_uid)
        count += len(batch_result)

        for re in batch_result:
            result["precision"] += re["precision"] / n_test_users
            result["recall"] += re["recall"] / n_test_users
            result["ndcg"] += re["ndcg"] / n_test_users
            result["hit_ratio"] += re["hit_ratio"] / n_test_users
            result["auc"] += re["auc"] / n_test_users
            result["f1"] += re["f1"] / n_test_users
            # f1_list.append(re["f1"])

    assert count == n_test_users
    # result["f1"] = np.mean(f1_list)  # 计算全局平均 F1
    pool.close()
    return result
