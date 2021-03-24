import matplotlib.pyplot as plt
import numpy as np
import tensorflow 

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                        100*np.max(predictions_array),
                                        class_names[true_label]),
                                        color=color)

def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

def full_plot(model):
    num_rows = 2
    num_cols = 5
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image(i, model[i], test_labels, test_images)
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_value_array(i, model[i], test_labels)
    plt.tight_layout()
    plt.show()

def plot_image2(predictions_array, true_label, img):
    true_label, img = true_label, img
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                        100*np.max(predictions_array),
                                        class_names[true_label]),
                                        color=color)
def plot_value_array2(predictions_array, true_label):
    true_label = true_label
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

def full_plot2(model, test_images, test_labels):
    num_rows = 2
    num_cols = 5
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    
    for i in range(num_images):
        nb_true = np.zeros(num_images)
        nb_total = np.zeros(num_images)
    
        for index in range(len(test_images)):
            if(test_labels[index] == i):
                nb_total += 1
                nb_true[np.argmax(model[index])] +=1

        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image2(nb_true/nb_total, test_labels[test_labels == i][0], test_images[test_labels == i][0])
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_value_array2(nb_true/nb_total, test_labels[test_labels == i][0])
    
    plt.tight_layout()
    plt.show()

def plot_name2(all_logs, exp, name="loss", ax = None):
    for k in range(len(all_logs)):
        res = [[] for _ in range(len(all_logs[k]))]
        for i in range(len(all_logs[k])):
            res[i] = all_logs[k][i].history[name]
        mean = np.mean(res, axis = 0)
        std = np.std(res, axis = 0)  
        if ax is None:
            ax = plt.gca()
        ax.plot(list(range(len(mean))), mean, label = exp[k])
        ax.fill_between(list(range(len(mean))), mean - std, mean + std, alpha = 0.2 )
    ax.set_title("mean " + name)
    ax.set_xlabel("epoch")
    return ax

def plot_logs2(all_logs, exp):
    f, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2, 2, sharey='row', figsize=(10,10))
    p1 = plot_name2(all_logs, name = 'loss' , exp = exp, ax = ax1)
    p2 = plot_name2(all_logs, name = 'val_loss' , exp = exp, ax = ax2)
    p3 = plot_name2(all_logs, name = 'accuracy' , exp = exp, ax = ax3)
    p4 = plot_name2(all_logs, name = 'val_accuracy' , exp = exp, ax = ax4)
    p2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    p4.legend(loc='center left', bbox_to_anchor=(1, 0.5))

