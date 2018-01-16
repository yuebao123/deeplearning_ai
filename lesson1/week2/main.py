
# from PIL import Image
import numpy as np
from scipy import ndimage, misc
import tensorflow as tf
import matplotlib.pyplot as plt
import h5py
from PIL import Image
from scipy import ndimage
import sklearn
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers import Dense, Activation

from week2.lr_utils import load_dataset, load_dataset1


def read_image(num_px, image_name):
    my_image = image_name
    fname = "images/" + my_image
    image_data = ndimage.imread(fname, flatten=False)
    image = np.array(image_data)
    image_x = misc.imresize(image, size=(num_px, num_px)).reshape((1, num_px * num_px * 3)).T
    return image_x, image


def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0
    return w, b


def propagate(w, b, X, Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)
    cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    dw = 1 / m * np.dot(X, (A - Y).T)
    db = 1 / m * np.sum(A - Y)
    cost = np.squeeze(cost)
    grads = {
        "dw": dw,
        "db": db,
    }
    return cost, grads


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    costs = []
    dw = 0
    db = 0
    for i in range(num_iterations):
        cost, grads = propagate(w, b, X, Y)
        dw = grads['dw']
        db = grads['db']
        w -= learning_rate * dw
        b -= learning_rate * db
        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print("Cost after iteration {}: {}".format(i, cost))

    params = {
        "w": w,
        "b": b
    }

    grads = {
        "dw": dw,
        "db": db
    }

    return params, grads, costs


def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    A = sigmoid(np.dot(w.T, X) + b)
    for i in range(A.shape[1]):
        if A[0, i] <= 0.5:
            Y_prediction[0, i] = 0
        else:
            Y_prediction[0, i] = 1

    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    w, b = initialize_with_zeros(X_train.shape[0])
    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    w = params['w']
    b = params['b']
    Y_prediction_train = predict(w, b, X_train)
    Y_prediction_test = predict(w, b, X_test)
    print("train accuracy: {}".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {}".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {
        "costs": costs,
        "Y_prediction_test": Y_prediction_test,
        "Y_prediction_train": Y_prediction_train,
        "w": w,
        "b": b,
        "learning_rate": learning_rate,
        "num_iterations": num_iterations,
    }

    return d


def use_manual():
    # learning_rates = [0.01, 0.001, 0.0001]
    learning_rates = [0.005]
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
    num_px = train_set_x_orig.shape[1]
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
    train_set_x = train_set_x_flatten / 255
    test_set_x = test_set_x_flatten / 255
    d = {}
    for learning_rate in learning_rates:
        d[str(learning_rate)] = model(train_set_x, train_set_y, test_set_x, test_set_y,
                                      num_iterations=2000, learning_rate=learning_rate, print_cost=False)
    # for k, v in d.items():
    #     plt.plot(np.squeeze(v['costs']), label=k)
    #
    # plt.ylabel('cost')
    # plt.xlabel('iterations')
    #
    # legend = plt.legend(loc='upper center', shadow=True)
    # frame = legend.get_frame()
    # frame.set_facecolor('0.90')
    # plt.show()
    image_x, image = read_image(num_px, '3.jpg')
    my_predicted_image = predict(d[str(learning_rate)]["w"], d[str(learning_rate)]["b"], image_x)

    plt.imshow(image)
    plt.title("y = {}, your algorithm predicts a {} picture.".format(
        str(np.squeeze(my_predicted_image)), classes[int(np.squeeze(my_predicted_image)), ].decode("utf-8")))
    plt.show()


def use_sklearn():
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
    num_px = train_set_x_orig.shape[1]
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
    train_set_x = train_set_x_flatten / 255
    test_set_x = test_set_x_flatten / 255
    # lr = LogisticRegression(C=1000.0, random_state=0)
    lr = LogisticRegression()
    lr.fit(train_set_x.T, train_set_y.T)
    # lr.fit(test_set_x.T, test_set_y.T)
    print("train accuracy = {}".format(lr.score(train_set_x.T, train_set_y.T)*100))
    print("test accuracy = {}".format(lr.score(test_set_x.T, test_set_y.T)*100))
    image_x, image = read_image(num_px, '4.jpg')
    result = lr.predict(image_x.T)
    print(result)
    plt.imshow(image)
    y = result[0]
    plt.title("y = {}, your algorithm predicts a {} picture.".format(str(y), classes[int(y), ].decode("utf-8")))
    plt.show()


def use_keras():
    model = Sequential()


def use_tensorflow():
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
    num_train = train_set_x_orig.shape[0]
    num_px = train_set_x_orig.shape[1]
    num_test = train_set_x_orig.shape[0]
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
    train_set_x = train_set_x_flatten / 255
    test_set_x = test_set_x_flatten / 255

    # 参数定义
    learning_rate = 0.005
    training_epoch = 25
    display_step = 1

    x = tf.placeholder(tf.float32, [None, train_set_x.shape[0]])
    y = tf.placeholder(tf.float32, [None, train_set_y.shape[0]])

    # 变量定义
    W = tf.Variable(tf.zeros([train_set_x.shape[0], 1]))
    b = tf.Variable(tf.zeros([1]))

    # 计算预测值
    pred = tf.nn.sigmoid(tf.matmul(x, W) + b)
    # 计算损失值 使用相对熵计算损失值
    # cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))
    cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred + 1e-7) + (1 - y) * tf.log(1 - pred + 1e-7), reduction_indices=1))
    # 定义优化器
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    # 初始化所有变量值
    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)
        # for epoch in range(training_epoch):
        avg_cost = 0.
        for i in range(1000):
            _, c = sess.run([optimizer, cost], feed_dict={x: train_set_x.T, y: train_set_y.T})
            avg_cost = c
        # if (epoch + 1) % display_step == 0:
        #     print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
        print("Optimization Finished!")

        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy for 3000 examples
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("Accuracy:", accuracy.eval({x: test_set_x.T, y: test_set_y.T}))


def main():
    # use_manual()
    use_sklearn()


if __name__ == '__main__':
    main()
