#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>
#include <dirent.h>


#define INPUT_SIZE 785
#define LEARNING_RATE 0.001
#define EPOCHS 150

// Function Prototypes
void read_mnist_images(const char *filename, float **d_images, int *num_d_images, int *drows, int *dcols);
void read_mnist_labels(const char *filename, int **d_labels, int *num_d_labels);
void read_images(const char *filename, float **images, int num_samples, int *rows, int *cols);
void read_labels(const char *filename, int **labels, int num_samples);
void save_to_file(const char *d_image_file, const char *d_label_file,float *d_images, int *d_labels, int num_samples, int drows, int dcols);
void gradient_descent(float *images, int *labels, float *parametreler, int num_samples, const char *filename, const char *filenamew);
void sgd(float *images, int *labels, float *weights, int num_samples, const char *filename, const char *filenamew);
void adam(float *images, int *labels, float *weights, int num_samples, const char *filename, const char *filenamew);
float compute_loss(float *images, int *labels, float *weights, int num_samples);
float predict(float *weights, float *sample);
int sign(float x);
void read_mnist_images(const char *filename, float **d_images, int *num_d_images, int *drows, int *dcols);
void read_mnist_labels(const char *filename, int **d_labels, int *num_d_labels);
void save_to_file(const char *d_image_file, const char *d_label_file,float *d_images, int *d_labels, int num_samples, int drows, int dcols);



int main() {

    // Folder Name for results
    const char *folder_name = "../results";
    
    // Check if the folder exists
    DIR *dir = opendir(folder_name);
    if (dir) {
        closedir(dir); 
    } else {
        //Create the folder
        printf("The '%s' folder does not exist, creating it...\n", folder_name);
        if (mkdir(folder_name) == 0) {
            printf("The '%s' folder was successfully created.\n", folder_name);
        } else {
            perror("Failed to create the folder");
        }
    } 

    // File paths for training and testing data
	const char *d_image_file = "../mnist/train-images.idx3-ubyte";
	const char *d_label_file = "../mnist/train-labels.idx1-ubyte";
	const char *output_d_image_file = "../results/1our_images.dat";
	const char *output_d_label_file = "../results/1our_labels.dat";
	const char *test_d_image_file = "../mnist/t10k-images.idx3-ubyte";
	const char *test_d_label_file = "../mnist/t10k-labels.idx1-ubyte";
	const char *test_output_d_image_file = "../results/1test_our_images.dat";
	const char *test_output_d_label_file = "../results/1test_our_labels.dat";



    // Variables to hold image data and labels for MNIST dataset
    float *d_images = NULL;
    int *d_labels = NULL;
    int num_d_images = 0, num_d_labels = 0, drows = 0, dcols = 0;


    // Read training data (images and labels)
    read_mnist_images(d_image_file, &d_images, &num_d_images, &drows, &dcols);
    read_mnist_labels(d_label_file, &d_labels, &num_d_labels);

    int max_per_class = 200;
    int class1_count = 0, class3_count = 0;
    int d_input_size = drows * dcols;
    int total_samples = 2 * max_per_class;
    int i,k;

	// Allocate memory for images and labels for a custom dataset
    float *burcu_d_images = (float *)malloc(total_samples * d_input_size * sizeof(float));
    int *burcu_d_labels = (int *)malloc(total_samples * sizeof(int));
    if (!burcu_d_images || !burcu_d_labels) {
        printf("Error: Memory allocation failed.\n");
        exit(1);
    }

    int index = 0;
    // Prepare the dataset with classes 1 (A) and 0 (B)
    for (i = 0; i < num_d_images; i++) {
        if (d_labels[i] == 1 && class1_count < max_per_class) {
            // CLASS A => label 1
            memcpy(&burcu_d_images[index * d_input_size], &d_images[i * d_input_size], d_input_size * sizeof(float));
            burcu_d_labels[index] = 1; // Class A (label 1)
            index++;
            class1_count++;
        } else if (d_labels[i] == 0 && class3_count < max_per_class) {
            // CLASS B => label -1
            memcpy(&burcu_d_images[index * d_input_size], &d_images[i * d_input_size], d_input_size * sizeof(float));
            burcu_d_labels[index] = -1; // Class B (label -1)
            index++;
            class3_count++;
        }
        if (class1_count == max_per_class && class3_count == max_per_class) {
            break;
        }
    }

    // Save the custom dataset to file
    save_to_file(output_d_image_file, output_d_label_file, burcu_d_images, burcu_d_labels, index, drows, dcols);
    printf("Data saved to %s and %s\n", output_d_image_file, output_d_label_file);

    // Free memory allocated for MNIST dataset
    free(d_images);
    free(d_labels);
    d_images = NULL;
    d_labels = NULL;
    num_d_images = 0;
    num_d_labels = 0;
    drows = 0;
    dcols = 0;

    //CREATE TEST DATA
    //CREATE TEST DATA
    //CREATE TEST DATA
    //CREATE TEST DATA


    // Read test data
    read_mnist_images(test_d_image_file, &d_images, &num_d_images, &drows, &dcols);
    read_mnist_labels(test_d_label_file, &d_labels, &num_d_labels);

    max_per_class = 50;
    class1_count = 0, class3_count = 0;
    d_input_size = drows * dcols;
    total_samples = 2 * max_per_class;

    float *test_burcu_d_images = (float *)malloc(total_samples * d_input_size * sizeof(float));
    int *test_burcu_d_labels = (int *)malloc(total_samples * sizeof(int));
    if (!test_burcu_d_images || !test_burcu_d_labels) {
        printf("Error: Memory allocation failed.\n");
        exit(1);
    }

    index = 0;
    // Prepare the test dataset with classes 1 (A) and 0 (B)
    for (i = 0; i < num_d_images; i++) {
        if (d_labels[i] == 1 && class1_count < max_per_class) {
            // 1 => A CLASS
            memcpy(&test_burcu_d_images[index * d_input_size], &d_images[i * d_input_size], d_input_size * sizeof(float));
            test_burcu_d_labels[index] = 1;  // Class A (label 1) -> output 1
            index++;
            class1_count++;
        } else if (d_labels[i] == 0 && class3_count < max_per_class) {
            // 0 => B CLASS
            memcpy(&test_burcu_d_images[index * d_input_size], &d_images[i * d_input_size], d_input_size * sizeof(float));
            test_burcu_d_labels[index] = -1; // Class B (label -1) -> output -1
            index++;
            class3_count++;
        }
        if (class1_count == max_per_class && class3_count == max_per_class) {
            break;
        }
    }

    // Save the test dataset to file
    save_to_file(test_output_d_image_file, test_output_d_label_file, test_burcu_d_images, test_burcu_d_labels, index, drows, dcols);
    printf("Data saved to %s and %s\n", test_output_d_image_file, test_output_d_label_file);

    // Free memory allocated for test dataset
    free(d_images);
    free(d_labels);
    free(burcu_d_images);
    free(burcu_d_labels);
    free(test_burcu_d_images);
    free(test_burcu_d_labels);

    //OPTIMIZATION
    //OPTIMIZATION
    //OPTIMIZATION
    //OPTIMIZATION
    //OPTIMIZATION
    //OPTIMIZATION

    // Optimization Section (Gradient Descent, SGD, Adam)
    const char *image_file = "../results/1our_images.dat";
    const char *label_file = "../results/1our_labels.dat";
    const char *test_image_file = "../results/1test_our_images.dat";
    const char *test_label_file = "../results/1test_our_labels.dat";


    float *images = NULL;
    float *timages = NULL;
    int *tlabels = NULL;
    int *labels = NULL;
    int num_samples = 400, rows = 28, cols = 28, tnum_samples = 100;

	// Read training and test data
    read_images(image_file, &images, num_samples, &rows, &cols);
    read_labels(label_file, &labels, num_samples);

    read_images(test_image_file, &timages, tnum_samples, &rows, &cols);
    read_labels(test_label_file, &tlabels, tnum_samples);

	// Initialize weights for optimization algorithms
    float weights_gd[INPUT_SIZE];
    float weights_sgd[INPUT_SIZE];
    float weights_adam[INPUT_SIZE];

    char wgd_filename[500];
    char wsgd_filename[500];
    char wadam_filename[500];

    char gd_filename[500];
    char sgd_filename[500];
    char adam_filename[500];

    // Initialize different weight sets for experiments
    float initial_weights[5][INPUT_SIZE] = {
        {0},
        {0},
        {0},
        {0},
        {0}
    };

	// Set initial weight values
    for(i=0;i<INPUT_SIZE;i++){
        initial_weights[0][i]= 0; //1. weights set
        initial_weights[1][i]= 0.01; //2. weights set
        initial_weights[2][i]= 0.005; //3. weights set
        initial_weights[3][i]= -0.005; //4. weights set
        initial_weights[4][i]= -0.01;  //5. weights set
    }

    for ( i = 0; i < 5; i++) {
        printf("\nTraining with Weight %d\n", i + 1);

        sprintf(gd_filename, "C:/Users/Lenovo/Desktop/Project-Optimization-Comparison/results/gd_results_%d.csv", i + 1);
        sprintf(sgd_filename, "C:/Users/Lenovo/Desktop/Project-Optimization-Comparison/results/sgd_results_%d.csv", i + 1);
        sprintf(adam_filename, "C:/Users/Lenovo/Desktop/Project-Optimization-Comparison/results/adam_results_%d.csv", i + 1);

        sprintf(wgd_filename, "C:/Users/Lenovo/Desktop/Project-Optimization-Comparison/results/w1_trajectory_gd_%d.csv", i + 1);
        sprintf(wsgd_filename, "C:/Users/Lenovo/Desktop/Project-Optimization-Comparison/results/w1_trajectory_sgd_%d.csv", i + 1);
        sprintf(wadam_filename, "C:/Users/Lenovo/Desktop/Project-Optimization-Comparison/results/w1_trajectory_adam_%d.csv", i + 1);

	// Call optimization algorithms (Gradient Descent, SGD, Adam)
        // GD
        memcpy(weights_gd, initial_weights[i], sizeof(float) * INPUT_SIZE);
        gradient_descent(images, labels, weights_gd, num_samples, gd_filename, wgd_filename);

        // SGD
        memcpy(weights_sgd, initial_weights[i], sizeof(float) * INPUT_SIZE);
        sgd(images, labels, weights_sgd, num_samples, sgd_filename, wsgd_filename);

        // Adam
        memcpy(weights_adam, initial_weights[i], sizeof(float) * INPUT_SIZE);
        adam(images, labels, weights_adam, num_samples, adam_filename, wadam_filename);

        int GDmodelAccuracy=0,SGDmodelAccuracy=0, ADAMmodelAccuracy=0 ;

        for(k=0; k<tnum_samples ;k++){
            if( sign(predict(weights_gd,&images[ k * INPUT_SIZE]))== labels[k] ){
                GDmodelAccuracy++;
            }

            if( sign(predict(weights_sgd,&images[ k * INPUT_SIZE]))== labels[k] ){
                SGDmodelAccuracy++;
            }
            if( sign(predict(weights_adam,&images[ k * INPUT_SIZE]))== labels[k] ){
                ADAMmodelAccuracy++;
            }

        }
        printf("\n\n\n");
        printf("GD Accuracy(%d): %d\n",i+1, GDmodelAccuracy);
        printf("SGD Accuracy(%d): %d\n",i+1, SGDmodelAccuracy);
        printf("ADAM Accuracy(%d): %d\n",i+1, ADAMmodelAccuracy);


    }

	// Free dynamically allocated memory
    free(images);
    free(labels);
    free(timages);
    free(tlabels);


    return 0;
}


/**
 * @brief Reads MNIST images from a file.
 * @param filename The name of the file containing the image data.
 * @param d_images Pointer to a float pointer to store image data.
 * @param num_d_images The number of images to read.
 * @param drows The number of rows in each image.
 * @param dcols The number of columns in each image.
 */
void read_mnist_images(const char *filename, float **d_images, int *num_d_images, int *drows, int *dcols) {
	int i;
    FILE *file = fopen(filename, "rb");
    if (!file) {
        printf("Error: Could not open file %s\n", filename);
        exit(1);
    }

    int32_t magic_number = 0, n_d_images = 0, n_drows = 0, n_dcols = 0;

    // Read metadata (magic number, number of images, rows, columns)
    fread(&magic_number, sizeof(int32_t), 1, file);
    fread(&n_d_images, sizeof(int32_t), 1, file);
    fread(&n_drows, sizeof(int32_t), 1, file);
    fread(&n_dcols, sizeof(int32_t), 1, file);

    // Convert from big-endian to little-endian
    magic_number = __builtin_bswap32(magic_number);
    n_d_images = __builtin_bswap32(n_d_images);
    n_drows = __builtin_bswap32(n_drows);
    n_dcols = __builtin_bswap32(n_dcols);

    *num_d_images = n_d_images;
    *drows = n_drows;
    *dcols = n_dcols;

    // Allocate memory for image data
    *d_images = (float *)malloc(n_d_images * n_drows * n_dcols * sizeof(float));
    if (*d_images == NULL) {
        printf("Error: Memory allocation failed for d_images.\n");
        exit(1);
    }

    // Read pixel values and normalize them
    for (i = 0; i < n_d_images * n_drows * n_dcols; i++) {
        unsigned char pixel = 0;
        fread(&pixel, sizeof(unsigned char), 1, file);
        (*d_images)[i] = pixel / 255.0f; // Normalize to [0, 1]
    }

    fclose(file);
}

/**
 * @brief Reads MNIST labels from a file.
 * @param filename The name of the file containing the label data.
 * @param d_labels Pointer to an integer pointer to store label data.
 * @param num_d_labels The number of labels to read.
 */
void read_mnist_labels(const char *filename, int **d_labels, int *num_d_labels) {
	int i;
    FILE *file = fopen(filename, "rb");
    if (!file) {
        printf("Error: Could not open file %s\n", filename);
        exit(1);
    }

    int32_t magic_number = 0, n_d_labels = 0;

    // Read metadata (magic number, number of labels)
    fread(&magic_number, sizeof(int32_t), 1, file);
    fread(&n_d_labels, sizeof(int32_t), 1, file);

    // Convert from big-endian to little-endian
    magic_number = __builtin_bswap32(magic_number);
    n_d_labels = __builtin_bswap32(n_d_labels);

    *num_d_labels = n_d_labels;

    // Allocate memory for label data
    *d_labels = (int *)malloc(n_d_labels * sizeof(int));
    if (*d_labels == NULL) {
        printf("Error: Memory allocation failed for d_labels.\n");
        exit(1);
    }

    // Read labels
    for (i = 0; i < n_d_labels; i++) {
        unsigned char label = 0;
        fread(&label, sizeof(unsigned char), 1, file);
        (*d_labels)[i] = label;
    }

    fclose(file);
}

/**
 * @brief Saves the our new custom dataset to a new file.
 * @param d_image_file The file to save the image data.
 * @param d_label_file The file to save the label data.
 * @param d_images The image data to save.
 * @param d_labels The label data to save.
 * @param num_samples The number of samples in the dataset.
 * @param drows The number of rows in each image.
 * @param dcols The number of columns in each image.
 */
void save_to_file(const char *d_image_file, const char *d_label_file,
                  float *d_images, int *d_labels, int num_samples, int drows, int dcols) {
    FILE *img_file = fopen(d_image_file, "wb");
    FILE *lbl_file = fopen(d_label_file, "wb");

    if (!img_file || !lbl_file) {
        printf("Error: Could not open files for writing.\n");
        exit(1);
    }

    fwrite(&num_samples, sizeof(int), 1, img_file); //Write number of samples
    fwrite(&drows, sizeof(int), 1, img_file); //Write image size rows
    fwrite(&dcols, sizeof(int), 1, img_file); //Write image size cols

    fwrite(d_images, sizeof(float), num_samples * drows * dcols, img_file); //Write image data
    fwrite(d_labels, sizeof(int), num_samples, lbl_file); //Write label data

    fclose(img_file);
    fclose(lbl_file);
}

/**
 * @brief Reads the our new custom images from a file.
 * @param filename The name of the file containing the image data.
 * @param images Pointer to a float pointer to store image data.
 * @param num_samples The number of samples in the dataset.
 * @param rows The number of rows in each image.
 * @param cols The number of columns in each image.
 */
void read_images(const char *filename, float **images, int num_samples, int *rows, int *cols) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        printf("Error: Could not open file %s\n", filename);
        exit(1);
    }

    *rows = 28;
    *cols = 28;

    int image_size = (*rows) * (*cols);
    *images = (float *)malloc((num_samples) * image_size * sizeof(float));
    if (!*images) {
        printf("Error: Memory allocation failed.\n");
        fclose(file);
        exit(1);
    }

    fread(*images, sizeof(float), (num_samples) * image_size, file);
    fclose(file);
}

/**
 * @brief Reads the our new custom labels from a file.
 * @param filename The name of the file containing the label data.
 * @param labels Pointer to an integer pointer to store label data.
 * @param num_samples The number of labels in the dataset.
 */
void read_labels(const char *filename, int **labels, int num_samples) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        printf("Error: Could not open file %s\n", filename);
        exit(1);
    }

    *labels = (int *)malloc((num_samples) * sizeof(int));
    if (!*labels) {
        printf("Error: Memory allocation failed.\n");
        fclose(file);
        exit(1);
    }

    fread(*labels, sizeof(int), num_samples, file);
    fclose(file);
}


/**
 * @brief Performs gradient descent to optimize the parameters.
 * @param images The image data.
 * @param labels The labels for the images.
 * @param parameters The parameters to be optimized.
 * @param num_samples The total number of samples in the dataset.
 * @param filename The file name to store the loss and time information.
 * @param filenamew The file name to store the optimized parameters.
 */
void gradient_descent(float *images, int *labels, float *parameters, int num_samples, const char *filename, const char *filenamew) {
    int i, j, epoch;
    FILE *gd_file = fopen(filename, "w");
    if (!gd_file) {
    	printf("Error: Could not open file %s for writing.\n", gd_file);
    	exit(1);
	}
    
    FILE *w1_file = fopen(filenamew, "w");
    
    // Check if files were successfully opened
    if (!gd_file || !w1_file) {
        printf("Error: Could not create file.\n");
        exit(1);
    }

    // Write headers for the CSV files
    fprintf(gd_file, "Epoch,Loss,Time\n");
    fprintf(w1_file, "Epoch,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10\n");

    clock_t epoch_start = clock();

    // Perform gradient descent for a predefined number of epochs
    for (epoch = 0; epoch < EPOCHS; epoch++) {

        float gradient[INPUT_SIZE] = {0};

        // Compute gradient for each sample
        for (i = 0; i < num_samples; i++) {
            float matris_carpim = 0;
            for (j = 0; j < INPUT_SIZE; j++) {
                matris_carpim += parameters[j] * images[i * INPUT_SIZE + j];
            }
            float error = labels[i] - tanh(matris_carpim);
            // Accumulate gradients for each parameter
            for (j = 0; j < INPUT_SIZE; j++) {
                gradient[j] += -2 * error * (1 - tanh(matris_carpim) * tanh(matris_carpim)) * images[i * INPUT_SIZE + j];
            }
        }

        // Update parameters using the computed gradients
        for (j = 0; j < INPUT_SIZE; j++) {
            parameters[j] -= LEARNING_RATE * gradient[j] / num_samples;
        }

        // Compute the loss function value
        float loss = compute_loss(images, labels, parameters, num_samples);

        // Save the current parameters to file
        fprintf(w1_file, "%d,%.6f", epoch + 1, parameters[0]);
        for (i = 1; i < INPUT_SIZE; i++) {
            fprintf(w1_file, ",%.6f", parameters[i]);
        }
        fprintf(w1_file, "\n");

        clock_t epoch_end = clock();
        float epoch_time = (float)(epoch_end - epoch_start) / CLOCKS_PER_SEC; // Calculate time in seconds

        // Print progress to console and log to file
        //printf("GD -> Epoch: %d, Loss: %.4f, Time: %.4f seconds\n", epoch + 1, loss, epoch_time);
        fprintf(gd_file, "%d,%.4f,%.4f\n", epoch + 1, loss, epoch_time); // Log to CSV file
        

    }

    // Close the files after writing
    fclose(gd_file);
    fclose(w1_file);
}

/**
 * @brief Performs stochastic gradient descent to optimize the model weights.
 * @param images The input image data.
 * @param labels The labels for the images.
 * @param weights The weights to be optimized during training.
 * @param num_samples The total number of samples in the dataset.
 * @param filename The file name to store the loss and time information.
 * @param filenamew The file name to store the optimized weights.
 */
void sgd(float *images, int *labels, float *weights, int num_samples, const char *filename, const char *filenamew) {
    int i, j, epoch;
    FILE *sgd_file = fopen(filename, "w");
    FILE *w1_file = fopen(filenamew, "w");

    // Check if files were successfully opened
    if (!sgd_file || !w1_file) {
        printf("Error: Could not create file.\n");
        exit(1);
    }

    // Write headers for the CSV files
    fprintf(sgd_file, "Epoch,Loss,Time\n");
    fprintf(w1_file, "Epoch,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10\n");

    clock_t epoch_start = clock();

    // Perform stochastic gradient descent for a predefined number of epochs
    for (epoch = 0; epoch < EPOCHS; epoch++) {

        // Compute gradient and update weights for each sample
        for (i = 0; i < num_samples; i++) {
            float vector_product = 0;
            for (j = 0; j < INPUT_SIZE; j++) {
                vector_product += weights[j] * images[i * INPUT_SIZE + j];
            }
            float error = labels[i] - tanh(vector_product);

            // Update weights immediately after calculating error for each sample
            for (j = 0; j < INPUT_SIZE; j++) {
                weights[j] -= LEARNING_RATE * (-2 * error * (1 - tanh(vector_product) * tanh(vector_product)) * images[i * INPUT_SIZE + j]);
            }
        }

        // Compute and log the loss
        float loss = compute_loss(images, labels, weights, num_samples);

        // Save the current weights to file
        fprintf(w1_file, "%d,%.6f", epoch + 1, weights[0]);
        for (i = 1; i < INPUT_SIZE; i++) {
            fprintf(w1_file, ",%.6f", weights[i]);
        }
        fprintf(w1_file, "\n");

        clock_t epoch_end = clock();
        float epoch_time = (float)(epoch_end - epoch_start) / CLOCKS_PER_SEC;

        // Print progress and log to file
        //printf("SGD -> Epoch: %d, Loss: %.4f, Time: %.4f seconds\n", epoch + 1, loss, epoch_time);
        fprintf(sgd_file, "%d,%.4f,%.4f\n", epoch + 1, loss, epoch_time);
    }

    fclose(sgd_file);
    fclose(w1_file);
}

/**
 * @brief Performs the Adam optimization algorithm to optimize model weights.
 * @param images The input image data.
 * @param labels The labels for the images.
 * @param weights The weights to be optimized during training.
 * @param num_samples The total number of samples in the dataset.
 * @param filename The file name to store the loss and time information.
 * @param filenamew The file name to store the optimized weights.
 */
void adam(float *images, int *labels, float *weights, int num_samples, const char *filename, const char *filenamew) {
    int i, j, epoch;
    FILE *adam_file = fopen(filename, "w");
    FILE *w1_file = fopen(filenamew, "w");

    // Check if files were successfully opened
    if (!adam_file || !w1_file) {
        printf("Error: Could not create file.\n");
        exit(1);
    }

    // Write headers for the CSV files
    fprintf(adam_file, "Epoch,Loss,Time\n");
    fprintf(w1_file, "Epoch,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10\n");

    // Adam-specific parameters
    float m[INPUT_SIZE] = {0};
    float v[INPUT_SIZE] = {0};
    float beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8;

    clock_t epoch_start = clock();

    // Perform Adam optimization for a predefined number of epochs
    for (epoch = 0; epoch < EPOCHS; epoch++) {

        float gradient[INPUT_SIZE] = {0};

        // Compute gradients for each sample
        for (i = 0; i < num_samples; i++) {
            float matrix_product = 0;
            for (j = 0; j < INPUT_SIZE; j++) {
                matrix_product += weights[j] * images[i * INPUT_SIZE + j];
            }
            float error = labels[i] - tanh(matrix_product);
            for (j = 0; j < INPUT_SIZE; j++) {
                gradient[j] += -2 * error * (1 - tanh(matrix_product) * tanh(matrix_product)) * images[i * INPUT_SIZE + j];
            }
        }

        // Update weights using Adam's optimization rules
        for (j = 0; j < INPUT_SIZE; j++) {
            gradient[j] /= num_samples;
            m[j] = beta1 * m[j] + (1 - beta1) * gradient[j];
            v[j] = beta2 * v[j] + (1 - beta2) * gradient[j] * gradient[j];
            float m_hat = m[j] / (1 - pow(beta1, epoch + 1));
            float v_hat = v[j] / (1 - pow(beta2, epoch + 1));
            weights[j] -= LEARNING_RATE * m_hat / (sqrt(v_hat) + epsilon);
        }

        // Compute and log the loss
        float loss = compute_loss(images, labels, weights, num_samples);

        // Save the current weights to file
        fprintf(w1_file, "%d,%.6f", epoch + 1, weights[0]);
        for (i = 1; i < INPUT_SIZE; i++) {
            fprintf(w1_file, ",%.6f", weights[i]);
        }
        fprintf(w1_file, "\n");

        clock_t epoch_end = clock();
        float epoch_time = (float)(epoch_end - epoch_start) / CLOCKS_PER_SEC;

        // Print progress and log to file
        //printf("ADAM -> Epoch: %d, Loss: %.4f, Time: %.4f seconds\n", epoch + 1, loss, epoch_time);
        fprintf(adam_file, "%d,%.4f,%.4f\n", epoch + 1, loss, epoch_time);
    }

    fclose(adam_file);
    fclose(w1_file);
}

/**
 * @brief Computes the loss for a set of predictions.
 * @param images The input image data.
 * @param labels The true labels for the images.
 * @param weights The model weights.
 * @param num_samples The number of samples in the dataset.
 * @return The calculated loss.
 */
float compute_loss(float *images, int *labels, float *weights, int num_samples) {
    int i, j;
    float loss = 0;

    // Calculate the squared error loss for each sample
    for (i = 0; i < num_samples; i++) {
        float matrix_product = 0;
        for (j = 0; j < INPUT_SIZE; j++) {
            matrix_product += weights[j] * images[i * INPUT_SIZE + j];
        }
        float error = labels[i] - tanh(matrix_product);
        loss += error * error;
    }

    return loss / num_samples;
}

/**
 * @brief Makes a prediction for a given sample using the model weights.
 * @param weights The model weights.
 * @param sample The input sample.
 * @return The predicted label.
 */
float predict(float *weights, float *sample) {
    int j;
    float matrix_product = 0;

    // Compute the weighted sum of the input sample
    for (j = 0; j < INPUT_SIZE; j++) {
        matrix_product += weights[j] * sample[j];
    }

    return tanh(matrix_product);
}

/**
 * @brief Returns the sign of the given value.
 * @param x The value to check.
 * @return 1 if x is positive, -1 if x is negative, 0 if x is zero.
 */
int sign(float x){
    return x > 0 ? 1 : -1;
}

