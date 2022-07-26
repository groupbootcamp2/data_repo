csv_path = "..\\cifar10_100.csv"
images_dir_path= '..\\resources\\'
dataset_dir_name='..'
files_list_cifar10=[('data_batch_' + str(i)) for i in range(1, 6)] + ["test_batch"]
files_list_cifar100=['train', 'test']
meta_file_cifar10="batches.meta"
meta_file_cifar100="meta"
cifar10='cifar 10'
cifar100='cifar 100'
personal='personal'
label_head_cifar10=b'labels'
label_head_cifar100=b'coarse_labels'
image_size=32
num_classes_cifar10=10
num_classes_cifar100=15


train_part=0.6
validation_part=0.2
test_part=0.2

z_file_path='cifer10_100_0_1_2_4_14.npz'
personal_image_path="output1.png"

model_path="..\\model_cnn_cifar_bootcamp"
history_path="..\\history_model_cnn_cifar_bootcamp.csv"


my_class_predict=0
my_class_add_image=1
my_class_visu=2