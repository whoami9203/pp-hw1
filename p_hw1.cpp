#include <iostream>
#include <vector>
#include <algorithm>
#include <png.h>
#include <stdlib.h>
#include <stdio.h>
#include <chrono>
// #include <omp.h>
#include <pthread.h>

struct RGB {
    int r, g, b;
};

// Structure to pass arguments to threads
struct CumulateArgs {
    std::vector<std::vector<int>>* cumulativeSum;
    int tHeight;
    int tWidth;
    int plusPre;
    int hHeight;
    int hWidth;
    int height;
    int width;
    char channel;
    const std::vector<std::vector<RGB>>& inputImage;
};
struct FArags {
    std::vector<std::vector<int>>* cumulativeSum;
    std::vector<std::vector<int>>* temp;
    std::vector<std::vector<int>>* kernelSizes;
    int height;
    int width;
    int plusPre;
};
struct WArgs {
    int startRow;
    int endRow;
    int width;
    int row_size;
    png_bytep* row_pointers;
    const std::vector<std::vector<RGB>>& image;

    WArgs(int start, int end, int w, int rs, png_bytep* rp, const std::vector<std::vector<RGB>>& img)
        : startRow(start), endRow(end), width(w), row_size(rs), row_pointers(rp), image(img) {}
};

double calculateLuminance(const RGB& pixel) {
    return 0.299 * pixel.r + 0.587 * pixel.g + 0.114 * pixel.b;
}


int determineKernelSize(double brightness) {
    return brightness > 128 ? 10 : 5;
}

void applyFilterToChannel(
    const std::vector<std::vector<int>>& cumulativeInput, 
    std::vector<std::vector<int>>& output, 
    const std::vector<std::vector<int>>& kernelSizes, 
    int height,
    int width,
    int pre
) {
    height += pre;
    width += pre;

    for (int x = pre; x < height; x++) {
        for (int y = pre; y < width; y++) {
            int n = (kernelSizes[x-pre][y-pre] == 5) ? 25 : 121;
            int kernelRadius = (kernelSizes[x-pre][y-pre] >> 1);

            output[x-pre][y-pre] = (cumulativeInput[x + kernelRadius][y + kernelRadius]
                                  -cumulativeInput[x - kernelRadius - 1][y + kernelRadius]
                                  -cumulativeInput[x + kernelRadius][y - kernelRadius - 1]
                                  +cumulativeInput[x - kernelRadius - 1][y - kernelRadius - 1]) / n;
        }
    }
}

void* applyFilterToChannelThread(void* args) {
    FArags* threadArgs = (FArags*)args;
    applyFilterToChannel(*threadArgs->cumulativeSum, *threadArgs->temp, *threadArgs->kernelSizes,
                         threadArgs->height, threadArgs->width, threadArgs->plusPre);
    pthread_exit(NULL);
}

void* calculateKernelSizes(void* args) {
    auto inputImage = ((CumulateArgs*)args)->inputImage;
    std::vector<std::vector<int>>* kernelSizes = ((CumulateArgs*)args)->cumulativeSum;
    int height = ((CumulateArgs*)args)->height;
    int width = ((CumulateArgs*)args)->width;

    for (int x = 0; x < height; x++) {
        for (int y = 0; y < width; y++) {
            double brightness = calculateLuminance(inputImage[x][y]);
            (*kernelSizes)[x][y] = determineKernelSize(brightness);
        }
    }
    pthread_exit(NULL);
}

void* calculateCumulativeSum(void* args) {
    CumulateArgs* threadArgs = (CumulateArgs*)args;
    auto& inputImage = threadArgs->inputImage;
    std::vector<std::vector<int>>* cumulativeSum = threadArgs->cumulativeSum;

    int tHeight = threadArgs->tHeight;
    int tWidth = threadArgs->tWidth;
    int plusPre = threadArgs->plusPre;
    int hHeight = threadArgs->hHeight;
    int hWidth = threadArgs->hWidth;
    char channel = threadArgs->channel;

    if(channel == 'r'){
        for (int x = 1; x < tHeight; x++) {
            for (int y = 1; y < tWidth; y++) {
                int i = std::min(std::max(x, plusPre), hHeight - 1);
                int j = std::min(std::max(y, plusPre), hWidth - 1);

                (*cumulativeSum)[x][y] = inputImage[i - plusPre][j - plusPre].r - (*cumulativeSum)[x - 1][y - 1]
                                    + (*cumulativeSum)[x][y - 1] + (*cumulativeSum)[x - 1][y];
            }
        }
    }
    else if(channel == 'g'){
        for (int x = 1; x < tHeight; x++) {
            for (int y = 1; y < tWidth; y++) {
                int i = std::min(std::max(x, plusPre), hHeight - 1);
                int j = std::min(std::max(y, plusPre), hWidth - 1);

                (*cumulativeSum)[x][y] = inputImage[i - plusPre][j - plusPre].g - (*cumulativeSum)[x - 1][y - 1]
                                    + (*cumulativeSum)[x][y - 1] + (*cumulativeSum)[x - 1][y];
            }
        }
    }
    else{
        for (int x = 1; x < tHeight; x++) {
            for (int y = 1; y < tWidth; y++) {
                int i = std::min(std::max(x, plusPre), hHeight - 1);
                int j = std::min(std::max(y, plusPre), hWidth - 1);

                (*cumulativeSum)[x][y] = inputImage[i - plusPre][j - plusPre].b - (*cumulativeSum)[x - 1][y - 1]
                                    + (*cumulativeSum)[x][y - 1] + (*cumulativeSum)[x - 1][y];
            }
        }
    }
    
    pthread_exit(NULL);
}

void adaptiveFilterRGB(
    const std::vector<std::vector<RGB>>& inputImage,
    std::vector<std::vector<RGB>>& outputImage,
    int height, 
    int width
) {
    int plusDimension = 11, plusPre = 6;
    int tHeight = height+plusDimension, tWidth = width+plusDimension;
    int hHeight = height+plusPre, hWidth = width+plusPre;

    std::vector<std::vector<int>> kernelSizes(height, std::vector<int>(width));
    std::vector<std::vector<int>> redCumulativeSum(tHeight, std::vector<int>(tWidth));
    std::vector<std::vector<int>> greenCumulativeSum(tHeight, std::vector<int>(tWidth));
    std::vector<std::vector<int>> blueCumulativeSum(tHeight, std::vector<int>(tWidth));

    // #pragma omp parallel sections
    // {
    //     #pragma omp section
    //     {
    //         for (int x = 0; x < height; x++) {
    //             for (int y = 0; y < width; y++) {
    //                 double brightness = calculateLuminance(inputImage[x][y]);
    //                 kernelSizes[x][y] = determineKernelSize(brightness);
    //             }
    //         }
    //     }
    //     #pragma omp section
    //     {
    //         for (int x = 1; x < tHeight; x++) {
    //             for (int y = 1; y < tWidth; y++) {
    //                 int i = std::min(std::max(x, plusPre), hHeight - 1);
    //                 int j = std::min(std::max(y, plusPre), hWidth - 1);

    //                 redCumulativeSum[x][y] = inputImage[i-plusPre][j-plusPre].r - redCumulativeSum[x-1][y-1]
    //                                         + redCumulativeSum[x][y-1] + redCumulativeSum[x-1][y];
    //             }
    //         }
    //     }
    //     #pragma omp section
    //     {
    //         for (int x = 1; x < tHeight; x++) {
    //             for (int y = 1; y < tWidth; y++) {
    //                 int i = std::min(std::max(x, plusPre), hHeight - 1);
    //                 int j = std::min(std::max(y, plusPre), hWidth - 1);

    //                 greenCumulativeSum[x][y] = inputImage[i-plusPre][j-plusPre].g - greenCumulativeSum[x-1][y-1]
    //                                         + greenCumulativeSum[x][y-1] + greenCumulativeSum[x-1][y];
    //             }
    //         }
    //     }
    //     #pragma omp section
    //     {
    //         for (int x = 1; x < tHeight; x++) {
    //             for (int y = 1; y < tWidth; y++) {
    //                 int i = std::min(std::max(x, plusPre), hHeight - 1);
    //                 int j = std::min(std::max(y, plusPre), hWidth - 1);

    //                 blueCumulativeSum[x][y] = inputImage[i-plusPre][j-plusPre].b - blueCumulativeSum[x-1][y-1]
    //                                         + blueCumulativeSum[x][y-1] + blueCumulativeSum[x-1][y];
    //             }
    //         }
    //     }
    // }
    pthread_t threads[4]; // 4 threads: one for kernelSizes, and one for each color channel

    // Arguments for each thread
    CumulateArgs redArgs = { &redCumulativeSum, tHeight, tWidth, plusPre, hHeight, hWidth, height, width, 'r', inputImage };
    CumulateArgs greenArgs = { &greenCumulativeSum, tHeight, tWidth, plusPre, hHeight, hWidth, height, width, 'g', inputImage };
    CumulateArgs blueArgs = { &blueCumulativeSum, tHeight, tWidth, plusPre, hHeight, hWidth, height, width, 'b', inputImage };
    CumulateArgs kernelArgs = { &kernelSizes, 0, 0, 0, 0, 0, height, width, 'g', inputImage };

    // Create threads
    pthread_create(&threads[0], NULL, calculateKernelSizes, (void*)&kernelArgs);
    pthread_create(&threads[1], NULL, calculateCumulativeSum, (void*)&redArgs);
    pthread_create(&threads[2], NULL, calculateCumulativeSum, (void*)&greenArgs);
    pthread_create(&threads[3], NULL, calculateCumulativeSum, (void*)&blueArgs);

    // Wait for threads to finish
    for (int i = 0; i < 4; i++) {
        pthread_join(threads[i], NULL);
    }

    std::vector<std::vector<int>> tempRed(height, std::vector<int>(width));
    std::vector<std::vector<int>> tempGreen(height, std::vector<int>(width));
    std::vector<std::vector<int>> tempBlue(height, std::vector<int>(width));

    // #pragma omp parallel sections
    // {
    //     #pragma omp section
    //     {
    //         applyFilterToChannel(redCumulativeSum, tempRed, kernelSizes, height, width, plusPre);
    //     }
    //     #pragma omp section
    //     {
    //         applyFilterToChannel(greenCumulativeSum, tempGreen, kernelSizes, height, width, plusPre);
    //     }
    //     #pragma omp section
    //     {
    //         applyFilterToChannel(blueCumulativeSum, tempBlue, kernelSizes, height, width, plusPre);
    //     }
    // }
    // // applyFilterToChannel(redCumulativeSum, tempRed, kernelSizes, height, width, plusPre);
    // // applyFilterToChannel(greenCumulativeSum, tempRed, kernelSizes, height, width, plusPre);
    // // applyFilterToChannel(blueCumulativeSum, tempRed, kernelSizes, height, width, plusPre);

    FArags redF = { &redCumulativeSum, &tempRed, &kernelSizes, height, width, plusPre };
    FArags greenF = { &greenCumulativeSum, &tempGreen, &kernelSizes, height, width, plusPre };
    FArags blueF = { &blueCumulativeSum, &tempBlue, &kernelSizes, height, width, plusPre };

    // Declare threads
    pthread_t Fthreads[3];

    // Create threads for red, green, and blue channels
    pthread_create(&Fthreads[0], NULL, applyFilterToChannelThread, (void*)&redF);
    pthread_create(&Fthreads[1], NULL, applyFilterToChannelThread, (void*)&greenF);
    pthread_create(&Fthreads[2], NULL, applyFilterToChannelThread, (void*)&blueF);

    // Wait for all threads to finish
    for (int i = 0; i < 3; i++) {
        pthread_join(Fthreads[i], NULL);
    }

    for (int x = 0; x < height; x++) {
        for (int y = 0; y < width; y++) {
            outputImage[x][y].r = tempRed[x][y];
            outputImage[x][y].g = tempGreen[x][y];
            outputImage[x][y].b = tempBlue[x][y];
        }
    }
}

void read_png_file(char* file_name, std::vector<std::vector<RGB>>& image) {
    FILE *fp = fopen(file_name, "rb");
    if (!fp) {
        std::cerr << "Error: Cannot open file " << file_name << std::endl;
        exit(EXIT_FAILURE);
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) {
        std::cerr << "Error: Cannot create PNG read structure" << std::endl;
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        std::cerr << "Error: Cannot create PNG info structure" << std::endl;
        png_destroy_read_struct(&png, nullptr, nullptr);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    if (setjmp(png_jmpbuf(png))) {
        std::cerr << "Error during PNG creation" << std::endl;
        png_destroy_read_struct(&png, &info, nullptr);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    png_init_io(png, fp);
    png_read_info(png, info);

    int width = png_get_image_width(png, info);
    int height = png_get_image_height(png, info);
    png_byte color_type = png_get_color_type(png, info);
    png_byte bit_depth = png_get_bit_depth(png, info);

    if(bit_depth == 16)
        png_set_strip_16(png);

    if(color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb(png);

    if(color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
        png_set_expand_gray_1_2_4_to_8(png);

    if(png_get_valid(png, info, PNG_INFO_tRNS))
        png_set_tRNS_to_alpha(png);

    if(color_type == PNG_COLOR_TYPE_RGB ||
       color_type == PNG_COLOR_TYPE_GRAY ||
       color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_filler(png, 0xFF, PNG_FILLER_AFTER);

    if(color_type == PNG_COLOR_TYPE_GRAY ||
       color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png);

    png_read_update_info(png, info);

    png_bytep* row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * height);
    for(int y = 0; y < height; y++) {
        row_pointers[y] = (png_byte*)malloc(png_get_rowbytes(png,info));
    }

    png_read_image(png, row_pointers);

    fclose(fp);

    image.resize(height, std::vector<RGB>(width));
    for (int y = 0; y < height; y++) {
        png_bytep row = row_pointers[y];
        for (int x = 0; x < width; x++) {
            png_bytep px = &(row[x * 4]);
            image[y][x].r = px[0];
            image[y][x].g = px[1];
            image[y][x].b = px[2];
        }
        free(row_pointers[y]);
    }
    free(row_pointers);

    png_destroy_read_struct(&png, &info, nullptr);
}

void* processRows(void* args) {
    WArgs* threadArgs = (WArgs*)args;

    for (int y = threadArgs->startRow; y < threadArgs->endRow; y++) {
        threadArgs->row_pointers[y] = (png_byte*)malloc(threadArgs->row_size);
        for (int x = 0; x < threadArgs->width; x++) {
            threadArgs->row_pointers[y][x * 3] = threadArgs->image[y][x].r;
            threadArgs->row_pointers[y][x * 3 + 1] = threadArgs->image[y][x].g;
            threadArgs->row_pointers[y][x * 3 + 2] = threadArgs->image[y][x].b;
        }
    }

    pthread_exit(NULL);
}

void write_png_file(char* file_name, std::vector<std::vector<RGB>>& image) {
    int width = image[0].size();
    int height = image.size();

    FILE *fp = fopen(file_name, "wb");
    if (!fp) {
        std::cerr << "Error: Cannot open file " << file_name << std::endl;
        exit(EXIT_FAILURE);
    }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) {
        std::cerr << "Error: Cannot create PNG write structure" << std::endl;
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        std::cerr << "Error: Cannot create PNG info structure" << std::endl;
        png_destroy_write_struct(&png, nullptr);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    if (setjmp(png_jmpbuf(png))) {
        std::cerr << "Error during PNG creation" << std::endl;
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    png_init_io(png, fp);

    png_set_IHDR(
        png,
        info,
        width, height,
        8,
        PNG_COLOR_TYPE_RGB,
        PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT,
        PNG_FILTER_TYPE_DEFAULT
    );
    png_write_info(png, info);

    size_t row_size = png_get_rowbytes(png, info);

    int num_threads = 8;  // Set the number of threads you want
    pthread_t threads[num_threads];

    png_bytep* row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * height);
    int rows_per_thread = height / num_threads;

    // Create threads and distribute rows among them
    for (int i = 0; i < num_threads; i++) {
        int startRow = i * rows_per_thread;
        int endRow = (i == num_threads - 1) ? height : (i + 1) * rows_per_thread;

        WArgs* threadArgs = new WArgs(startRow, endRow, width, row_size, row_pointers, image);

        pthread_create(&threads[i], NULL, processRows, (void*)threadArgs);
    }

    // Wait for all threads to finish
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    // auto start = std::chrono::high_resolution_clock::now();

    png_write_image(png, row_pointers);
    png_write_end(png, nullptr);

    // auto write_end = std::chrono::high_resolution_clock::now();

    for (int y = 0; y < height; y++) {
        free(row_pointers[y]);
    }
    free(row_pointers);

    // auto free_end = std::chrono::high_resolution_clock::now();

    png_destroy_write_struct(&png, &info);
    fclose(fp);

    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> write_seconds = write_end - start;
    // std::chrono::duration<double> free_seconds = free_end - write_end;
    // std::chrono::duration<double> end_all_seconds = end - start;
    // std::cout << "write time: " << write_seconds.count() * 1000.0 << "ms" << std::endl;
    // std::cout << "free time: " << free_seconds.count() * 1000.0 << "ms" << std::endl;
    // std::cout << "write file time: " << end_all_seconds.count() * 1000.0 << "ms" << std::endl;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <inputfile.png> <outputfile.png>" << std::endl;
        return -1;
    }

    auto start_all = std::chrono::high_resolution_clock::now();

    char* input_file = argv[1];
    char* output_file = argv[2];

    std::vector<std::vector<RGB>> inputImage;
    read_png_file(input_file, inputImage);

    int height = inputImage.size();
    int width = inputImage[0].size();

    std::vector<std::vector<RGB>> outputImage(height, std::vector<RGB>(width));

    auto start = std::chrono::high_resolution_clock::now();

    adaptiveFilterRGB(inputImage, outputImage, height, width);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Main Program Time: " << elapsed_seconds.count() * 1000.0 << " ms" << std::endl;

    write_png_file(output_file, outputImage);

    auto end_all = std::chrono::high_resolution_clock::now();
    elapsed_seconds = end_all - start_all;
    std::cout << "Total Program Time: " << elapsed_seconds.count() * 1000.0 << " ms" << std::endl;

    return 0;
}
