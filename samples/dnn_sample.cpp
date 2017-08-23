#include <iostream>
#include <fstream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


using namespace std;
using namespace cv;
using namespace cv::dnn;


const char* kAbout = "Sample of DNN module usage.";

const char* kOptions =
    "{ i  image         | <none> | image to process        }"
    "{ mt modelTxt      | <none> | prototxt                }"
    "{ mb modelBin      | <none> | Caffe model             }"
    "{ c  classes       | <none> | the list of class names }"
    "{ h ? help usage   |        | print help message      }";


/* Find best class for the blob (i. e. class with maximal probability) */
static void getMaxClass(const Mat &probBlob, int *classId, double *classProb)
{
    //reshape the blob to 1x1000 matrix
    Mat probMat = probBlob.reshape(1, 1);
    Point classNumber;
    minMaxLoc(probMat, NULL, classProb, NULL, &classNumber);
    *classId = classNumber.x;
}
static std::vector<String> readClassNames(
    const char *filename = "synset_words.txt")
{   
    std::vector<String> classNames;
    std::ifstream fp(filename);
    if (!fp.is_open())
    {
        std::cout << "File with classes labels not found: " <<
            filename << std::endl;
        exit(-1);
    }
    std::string name;
    while (!fp.eof())
    {
        std::getline(fp, name);
        if (name.length())
        {
            classNames.push_back(name.substr(name.find(' ') + 1));
        }
    }
    fp.close();
    return classNames;
}


int main(int argc, char** argv)
{
    // Parse command line
    CommandLineParser parser(argc, argv, kOptions);
    parser.about(kAbout);

    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }

    // Load image
    String imgName(parser.get<String>("image"));
    Mat img = imread(imgName);
    if (img.empty())
    {
        std::cout << "Can't load image for classification:" << std::endl;
        std::cout << "image: " << imgName << std::endl;
        return 0;
    }

    // Load model
    String modelTxt(parser.get<String>("modelTxt"));
    String modelBin(parser.get<String>("modelBin"));
    Net net = dnn::readNetFromCaffe(modelTxt, modelBin);
    if (net.empty())
    {
        std::cout << "Can't load network by using the following files:" << std::endl;
        std::cout << "prototxt:   " << modelTxt << std::endl;
        std::cout << "caffemodel: " << modelBin << std::endl;
        std::cout << "bvlc_googlenet.caffemodel can be downloaded here:" << std::endl;
        std::cout << "http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel" << std::endl;
        return 0;
    }

    // GoogLeNet accepts only 224x224 RGB-images
    // Scalar(104, 117, 123) - mean value across full training set
    Mat inputBlob = blobFromImage(img, 1, Size(224, 224),
        Scalar(104, 117, 123)); //Convert Mat to batch of images
    Mat prob;
    cv::TickMeter t;
    for (int i = 0; i < 10; i++)
    {
        net.setInput(inputBlob, "data"); //set the network input
        t.start();
        prob = net.forward("prob"); //compute output
        t.stop();
    }

    // Find the best class
    int classId;
    double classProb;
    string classes = parser.get<string>("classes");
    getMaxClass(prob, &classId, &classProb);
    std::vector<String> classNames = readClassNames(classes.c_str());
    std::cout << "Best class: #" << classId << " '" <<
        classNames.at(classId) << "'" << std::endl;
    std::cout << "Probability: " << classProb * 100 << "%" << std::endl;
    std::cout << "Time: " << (double)t.getTimeMilli() / t.getCounter() <<
        " ms (average from " << t.getCounter() << " iterations)" << std::endl;

    return 0;
}
