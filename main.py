import create_data as cd
import data_visualization as dv
#
#cd.load_cifar10()
#
#cifar10_100=cd.load_x_labels_from_cifar100([0,1,2,3,4])

cd.insert_personal_image_to_csv("my try dog9.png",5 )

dv.show_10_image_of_class(0, 'cifar10')

dv.show_classes_count()

dv.show_splited_classes_count()

#
# cifar10_100=cd.load_x_labels_from_cifar100([5,7])
# #
# dv.show_10_image_of_class(5,'cifar10')
# #
# dv.show_classes_count()
# # #
# # dv.show_splited_classes_count()
# #
# cifar10_100=cd.load_x_labels_from_cifar100([5,9])
#
# dv.show_10_image_of_class(9, 'cifar100')
#
# dv.show_10_image_of_class(20, 'cifar100')
# cd.insert_personal_image_to_csv("my try dog8.png",5 )
#
# dv.show_classes_count()
# dv.show_splited_classes_count()

#cifar10_100=cd.load_x_labels_from_cifar100([11,9, 12])


#dv.show_10_image_of_class(11, 'cifar100')
#
#dv.show_classes_count()
#
#dv.show_splited_classes_count()
#
#
#
#
# cd.save_to_zip()

