from tanuki_ml import generate_model
import tanuki_ml
import sys
import time

'''
python acc... 5

인자 1 = memory size
'''

start_total = time.time()

# 초기화
memory_size = int(sys.argv[1])
color_num = 3
scaler = 3
input_shape = (memory_size, 590//scaler, 1640//scaler, color_num)
resized_shape = (1640//scaler, 590//scaler)
batch_size = 10
epochs = 2
pool_size = (2, 2)

print("Accuracy test with memory size =", memory_size)
print("Read model from data in", "mem_is_{}.h5".format(memory_size))

model = generate_model(input_shape, pool_size)
model.compile(optimizer='Adam', loss='mean_squared_error', metrics = ['accuracy'])
model.load_weights("mem_is_{}.h5".format(memory_size),"r")

# Data load
X_test, y_test = tanuki_ml.read_set('/home/mary/ml', resized_shape)

X_test_t, y_test_t = tanuki_ml.give_time(X_test, y_test, memory_size = memory_size)
del(X_test)
del(y_test)

start_test = time.time()
loss_and_metrics = model.evaluate(X_test_t, y_test_t, batch_size, verbose = 1)
end_test = time.time()

print('Loss is {:.5f}, Accuracy is {:.5f}%'.format(loss_and_metrics[0],loss_and_metrics[1]*100))

end_total = time.time()

## 실험 데이터 저장
f=open("Test_result_mem_is_{}.txt".format(memory_size),'w')

# 총 걸린 시간
min, sec = divmod(end_total-start_total, 60)
f.write("Total run time : {}min {}sec\n".format(min,sec))

# 총 걸린 평가 시간
min, sec = divmod(end_test-start_test, 60)
f.write("Pure test time : {}min {}sec\n".format(min,sec))

# Loss and Accuracy
f.write('Loss is {}, Accuracy is {}%\n'.format(loss_and_metrics[0],loss_and_metrics[1]*100))

f.close()
