from tanuki_ml import generate_model
import tanuki_ml

# 초기화
memory_size = 3
color_num = 3
scaler = 3
input_shape = (memory_size, 590//scaler, 1640//scaler, color_num)
resized_shape = (1640//scaler, 590//scaler)
batch_size = 10
epochs = 2
pool_size = (2, 2)

model = generate_model(input_shape, pool_size)
model.compile(optimizer='Adam', loss='mean_squared_error')
model.load_weights("tanuki_network.h5","r")

# Data load
import tanuki_ml

X_test, y_test = tanuki_ml.read_set('/home/mary/ml/test', resized_shape)

X_test_t, y_test_t = tanuki_ml.give_time(X_test, y_test, memory_size = 3)
del X_test; del y_test;

loss_and_metrics = model.evaluate(X_test_t, y_test_t, batch_size, verbose = 1)

print('Loss is {:3f}, Accuracy is {:3f}'.format(loss_and_metrics[0],loss_and_metrics[1]*100))