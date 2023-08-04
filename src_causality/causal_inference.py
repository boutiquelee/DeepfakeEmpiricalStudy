import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras import Model
import time
import datetime

RESULT_DIR = '../results'
IMG_ROWS = 256
IMG_COLS = 256
IMG_COLOR = 3
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, IMG_COLOR)
MASK_SHAPE = (IMG_ROWS, IMG_COLS)
NUM_CLASSES = 2
CALSAL_STEP = 4
TEST_ONLY = 1

class causal_analyzer:

    UPSAMPLE_SIZE = 1
    INTENSITY_RANGE = 'mnist'
    REGULARIZATION = 'l1'
    ATTACK_SUCC_THRESHOLD = 0.99
    PATIENCE = 10
    COST_MULTIPLIER = 1.5,
    RESET_COST_TO_ZERO = True
    MASK_MIN = 0
    MASK_MAX = 1
    COLOR_MIN = 0
    COLOR_MAX = 1
    IMG_COLOR = 3
    SHUFFLE = True
    BATCH_SIZE = 64
    VERBOSE = 1
    RETURN_LOGS = True
    SAVE_LAST = False
    EPSILON = K.epsilon()
    EARLY_STOP = True
    EARLY_STOP_THRESHOLD = 0.99
    EARLY_STOP_PATIENCE = 2 * PATIENCE
    SAVE_TMP = False
    TMP_DIR = 'tmp'
    RAW_INPUT_FLAG = False

    SPLIT_LAYER = 21
    REP_N =512

    def __init__(self, model, train_generator,test_generator, input_shape,
                 init_cost, steps, mini_batch, lr, num_classes,
                 upsample_size=UPSAMPLE_SIZE,
                 patience=PATIENCE, cost_multiplier=COST_MULTIPLIER,
                 reset_cost_to_zero=RESET_COST_TO_ZERO,
                 mask_min=MASK_MIN, mask_max=MASK_MAX,
                 color_min=COLOR_MIN, color_max=COLOR_MAX, img_color=IMG_COLOR,
                 shuffle=SHUFFLE, batch_size=BATCH_SIZE, verbose=VERBOSE,
                 return_logs=RETURN_LOGS, save_last=SAVE_LAST,
                 epsilon=EPSILON,
                 early_stop=EARLY_STOP,
                 early_stop_threshold=EARLY_STOP_THRESHOLD,
                 early_stop_patience=EARLY_STOP_PATIENCE,
                 save_tmp=SAVE_TMP, tmp_dir=TMP_DIR,
                 raw_input_flag=RAW_INPUT_FLAG,
                 rep_n=REP_N):

        self.model = model
        self.input_shape = input_shape
        self.train_gen=train_generator
        self.test_gen = test_generator
        self.init_cost = init_cost
        self.steps = 1
        self.mini_batch = mini_batch
        self.lr = lr
        self.num_classes = num_classes
        self.upsample_size = upsample_size
        self.patience = patience
        self.cost_multiplier_up = cost_multiplier
        self.cost_multiplier_down = cost_multiplier ** 1.5
        self.reset_cost_to_zero = reset_cost_to_zero
        self.mask_min = mask_min
        self.mask_max = mask_max
        self.color_min = color_min
        self.color_max = color_max
        self.img_color = img_color
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.verbose = verbose
        self.return_logs = return_logs
        self.save_last = save_last
        self.epsilon = epsilon
        self.early_stop = early_stop
        self.early_stop_threshold = early_stop_threshold
        self.early_stop_patience = early_stop_patience
        self.save_tmp = save_tmp
        self.tmp_dir = tmp_dir
        self.raw_input_flag = raw_input_flag
        self.rep_n = rep_n
        self.r_weight = None
        self.target = 1
        self.alpha = 0
        self.model1, self.model2 = self.split_keras_model(self.model, self.SPLIT_LAYER)

        pass

    def split_keras_model(self, lmodel, index):

        model1 = Model(inputs=lmodel.model.inputs, outputs=lmodel.model.layers[index - 1].output)
        model2_input = Input(lmodel.model.layers[index].input_shape[1:])
        model2 = model2_input

        for layer in lmodel.model.layers[index:]:
            model2 = layer(model2)

        model2 = Model(inputs=model2_input, outputs=model2)

        return (model1, model2)

    def analyze(self,train_gen, test_gen):
        alpha_list = [0.1]

        for alpha in alpha_list:
            self.alpha = alpha

            for i in range(0, 1):
                print('iteration: {}'.format(i))
                self.analyze_each(train_gen,test_gen)

    def analyze_each(self, train_gen,test_gen):
        ana_start_t = time.time()

        for step in range(self.steps):
            min_test = []
            min_train = []
            max_test = []
            max_train = []

            for idx in range(self.mini_batch):
                X_test_batch, Y_test_batch = test_gen.next()
                X_train_batch, Y_train_batch = train_gen.next()

                min_i, max_i = self.get_h_range(X_test_batch)
                min_test.append(min_i)
                max_test.append(max_i)

                min_i, max_i = self.get_h_range(X_train_batch)
                min_train.append(min_i)
                max_train.append(max_i)

                train_prediction = self.model.predict(X_train_batch)
                test_predict = self.model.predict(X_test_batch)
                np.savetxt("../results/train_prediction.txt", train_prediction, fmt="%s")
                np.savetxt("../results/test_predict.txt", test_predict, fmt="%s")

            min_train = np.min(np.array(min_train), axis=0)
            max_train = np.max(np.array(max_train), axis=0)

        for step in range(self.steps):
            ie_batch = []

            for idx in range(self.mini_batch):
                X_train_batch, _ = train_gen.next()
                ie_batch.append(self.get_tie_do_h(X_train_batch, np.minimum(min_train, 1), np.maximum(max_train, 0)))

            ie_mean = np.mean(np.array(ie_batch),axis=0)
            np.savetxt("../results/ori.txt", ie_mean, fmt="%s")

            col_diff = np.max(ie_mean, axis=0) - np.min(ie_mean, axis=0)
            col_diff = np.transpose([np.arange(len(col_diff)), col_diff])
            ind = np.argsort(col_diff[:, 1])[::-1]
            col_diff = col_diff[ind]
            np.savetxt("../results/col_diff.txt", col_diff, fmt="%s")

            row_diff = np.max(ie_mean, axis=1) - np.min(ie_mean, axis=1)
            row_diff = np.transpose([np.arange(len(row_diff)), row_diff])
            ind = np.argsort(row_diff[:, 1])[::-1]
            row_diff = row_diff[ind]
            np.savetxt("../results/row_diff.txt", row_diff, fmt="%s")

            ana_start_t = time.time() - ana_start_t
            print('fault localization time: {}s'.format(ana_start_t))
            rep_t = time.time()

            self.rep_index = []
            self.rep_index = row_diff[:,:1][:self.rep_n,:]
            np.set_printoptions(suppress=True, formatter={'int': '{:d}'.format})

            print("repair index: {}".format(self.rep_index.T))
            print(f"Number of elements in repair index: {np.size(self.rep_index)}")

            now = datetime.datetime.now()
            date_string = now.strftime("%Y-%m-%d_%H-%M-%S")
            with open(f"../results/neuron/{date_string}.txt", "w") as f:
                f.write(str(self.rep_index))
    pass

    def get_tie_do_h(self, x, min, max):
        pre_layer5 = self.model1.predict(x)
        l_shape = pre_layer5.shape
        ie = []
        hidden_min = min.reshape(-1)
        hidden_max = max.reshape(-1)
        num_step = CALSAL_STEP
        _pre_layer5 = np.reshape(pre_layer5, (len(pre_layer5), -1))

        for i in range (len(_pre_layer5[0])):
            ie_i = []
            for h_val in np.linspace(hidden_min[i], hidden_max[i], num_step):
                do_hidden = _pre_layer5.copy()
                do_hidden[:, i] = h_val
                pre_final = self.model2.predict(do_hidden.reshape(l_shape))
                ie_i.append(np.mean(pre_final, axis = 0))

            ie.append(np.array(ie_i))

        return np.array(ie).squeeze()

    def get_h_range(self, x):
        pre_layer5 = self.model1.predict(x)
        max = np.max(pre_layer5,axis=0)
        min = np.min(pre_layer5, axis=0)

        return min, max