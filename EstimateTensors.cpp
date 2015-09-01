
#include "iostream"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "GetPot/GetPot"
#include "inc/MapFilterLR2HR.h"
#include "itkDiffusionTensor3D.h"


using namespace std;

int main (int argc, char *argv[])
{

    GetPot cl (argc, const_cast<char**>(argv));
    if( cl.size() == 1 || cl.search (2,"--help","-h") )
    {
        std::cout << "Not Enough Arguments" << std::endl;
        std::cout << "Scales the tensors with a scalar factor" << std::endl;
        std::cout << "Usage: -trueB0 <true B0> -m <MaskImage> -true <True Tensors> -f <flag for extended gradient> -t <initial tensor estimate> -g <gradient> -o <Output File> -s <Sigma> -nm <Noise Model> -Sim <intelligent COnvergence>" << std::endl;
      return -1;
    }

    std::cout << "Starting Program " << std::endl;
    
    const string mask_file_N = cl.follow("NoFile", 1 , "-m");
    const string fileIn      = cl.follow("NoFile", 1 , "-i" );
    const string tensor_file_N = cl.follow("NoFile",1, "-t");
    const string grad_file_N = cl.follow ("NoFile", 1, "-g");
    const string B0_file_N  = cl.follow("NoFile", 1, "-B0");
    const string trans_n =cl.follow("NoFile",1,"-trans");

    // Typedefs
    typedef itk::Image<float, 3> ImageType;
    typedef itk::ImageFileReader<ImageType> ReaderType;
    typedef itk::ImageFileWriter<ImageType> WriterType;


    typedef itk::DiffusionTensor3D<float> DiffusionTensorType;
    typedef itk::Image<DiffusionTensorType, 3> TensorImageType;
    typedef itk::TransformFileReader TransformFileReaderType;
    typedef TransformFileReaderType::TransformListType TransformListType;
    typedef itk::TransformBase TransformBaseType;
    typedef itk::AffineTransform<double, 3> AffineTransformType;


    // Mask Image
    ReaderType::Pointer reader_mask = ReaderType::New();
    reader_mask ->SetFileName(mask_file_N);
    reader_mask->Update();
    ImageType::Pointer mask_Image = reader_mask->GetOutput();

    // B0 Image
    ReaderType::Pointer B0_image_reader = ReaderType::New();
    B0_image_reader->SetFileName(B0_file_N);
    B0_image_reader->Update();
    ImageType::Pointer B0_image = B0_image_reader->GetOutput();

    // Read Gradients
    typedef itk::Vector<float, 3> GradientType;
    typedef std::vector<GradientType> GradientListType;

       GradientListType GradientList;
       std::ifstream fileg(grad_file_N);
       int numOfGrads = 0;
       fileg >> numOfGrads;

       for (int i=0; i < numOfGrads; i++)
        {
            GradientType g;
            fileg >> g[0]; fileg >>g[1]; fileg >> g[2];
            if (g!=0.0)
            {
                g.Normalize();
            }
            GradientList.push_back(g);
//            std::cout << g << std::endl;
        }

      // Read Images
       typedef std::vector<ImageType::Pointer> ImageListType;
       ImageListType DWIList;

       std::ifstream file(fileIn);
       int numOfImages = 0;
       file >> numOfImages;

       for (int i=0; i < numOfImages  ; i++) // change of numOfImages
         {
             char filename[256];
             file >> filename;
             ReaderType::Pointer myReader=ReaderType::New();
             myReader->SetFileName(filename);
             std::cout << "Reading.." << filename << std::endl; // add a try catch block
             myReader->Update();
             DWIList.push_back( myReader->GetOutput() ); //using push back to create a stack of diffusion images
         }

       // Read tensor image
       typedef itk::ImageFileReader<TensorImageType> TensorReaderType;
       TensorReaderType::Pointer tensor_Reader = TensorReaderType::New();

       tensor_Reader->SetFileName(tensor_file_N);
       tensor_Reader->Update();

       TensorImageType::Pointer inital_tensor_Image = tensor_Reader->GetOutput();

       // Convert the TTK tensor to itk Tensor
       TensorImageType::IndexType testIndex;
       testIndex[0]=128; testIndex[1]=128; testIndex[2]=29;

//       DiffusionTensor
//
//       std::cout << inital_tensor_Image->GetPixel(testIndex) << std::endl;

       // Reading Transform
       TransformFileReaderType::Pointer readerTransform = TransformFileReaderType::New();
       	readerTransform->SetFileName(trans_n);
       	readerTransform -> Update();
       	TransformListType *list = readerTransform->GetTransformList();
       	TransformBaseType * transform = list->front().GetPointer();
       	TransformBaseType::ParametersType parameters = transform->GetParameters();
       	AffineTransformType::Pointer transform_fwd = AffineTransformType::New();
       	transform_fwd->SetParameters(parameters);

       	MapFilterLR2HR1 filter;
       	filter.ComputeMap(DWIList[0], B0_image, transform_fwd);

//       	vnl_sparse_matrix<float> MapLR2HR, MapHR2LR;

//       	MapLR2HR = filter.GetLR2HRMatrix();
//       	MapHR2LR = filter.GetHR2LRMatrix();
//



    return 0;
}
