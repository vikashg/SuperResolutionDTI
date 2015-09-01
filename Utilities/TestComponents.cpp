/*
 * TestComponents.cxx
 *
 *  Created on: Jul 29, 2015
 *      Author: vgupta
 */


#include "iostream"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkDiffusionTensor3D.h"
#include "../inc/TensorUtilites.h"
#include "vnl/vnl_matrix.h"
#include "JointTensorEstimation.h"
#include "../GetPot/GetPot"


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

    typedef float RealType;
	typedef itk::DiffusionTensor3D<RealType> DiffusionTensorType;
	typedef itk::Vector<RealType, 3 > VectorType;
	typedef vnl_matrix<RealType> MatrixType;

	VectorType g;

	const string file_g_n = cl.follow("NoFile",1, "-g");
	const string file_in = cl.follow("NoFile",1, "-i");
	const string mask_n  = cl.follow ("NoFile",  1 , "-m");
	const string tens_n = cl.follow("NoFile", 1, "-t");

	typedef itk::DiffusionTensor3D<RealType> DiffusionTensorType;
	typedef itk::Image<DiffusionTensorType, 3> TensorImageType;

	typedef itk::ImageFileReader<TensorImageType> ReaderType;
	ReaderType::Pointer tensorReader = ReaderType::New();

	tensorReader->SetFileName(tens_n);
	tensorReader->Update();
	TensorImageType::Pointer tensorImage = tensorReader ->GetOutput();

	typedef itk::Image<RealType, 3> ScalarImageType;
	ScalarImageType::Pointer maskImage = ScalarImageType::New();
	typedef itk::ImageFileReader<ScalarImageType> ScalarReaderType;
	ScalarReaderType::Pointer maskReader = ScalarReaderType::New();

	maskReader->SetFileName(mask_n);
	maskReader->Update();

	maskImage = maskReader->GetOutput();

	JointTensorEstimation jTestimation; 	jTestimation.ReadKappa(1);

	ScalarImageType::Pointer gradMagTensor = jTestimation.GradientLogMagTensorImage(tensorImage, maskImage);

	TensorUtilities utilsTens;

	TensorImageType::Pointer LogTensorImage = utilsTens.LogTensorImageFilter(tensorImage, maskImage);

	TensorImageType::Pointer LaplaceTensor = jTestimation.ComputeLaplaceTensor(LogTensorImage, maskImage);

	typedef itk::ImageFileWriter <ScalarImageType> WriterType;
	WriterType::Pointer scaWriter = WriterType::New();
	scaWriter->SetFileName("gradMagTensorImage.nii.gz");
	scaWriter->SetInput(gradMagTensor);
	scaWriter->Update();

	typedef itk::ImageFileWriter<TensorImageType> TensorWriterType;

	ScalarImageType::Pointer PsiImage = jTestimation.ComputePsiImage(gradMagTensor, maskImage);

	scaWriter->SetFileName("PsiImage.nii.gz");
	scaWriter->SetInput(PsiImage);
	scaWriter->Update();

	TensorImageType::Pointer first_term =jTestimation.ComputeFirstTermDelReg(PsiImage, LaplaceTensor, maskImage);
	TensorImageType::Pointer second_term = jTestimation.ComputeSecondTermDelReg(PsiImage, LogTensorImage, maskImage);

	TensorWriterType::Pointer tensorWriter = TensorWriterType::New();
	tensorWriter->SetFileName("FirstTerm.nii.gz");
	tensorWriter->SetInput(first_term);
	tensorWriter->Update();


	tensorWriter->SetFileName("SecondTerm.nii.gz");
	tensorWriter->SetInput(second_term);
	tensorWriter->Update();


	// Testing ComputDelsim()



	return 0;

}
