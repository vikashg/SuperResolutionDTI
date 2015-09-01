/*
 * TransformGradientTable.cxx
 *
 *  Created on: Jul 21, 2015
 *      Author: vgupta
 */

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkTransformToDisplacementFieldFilter.h"
#include "../GetPot/GetPot"
#include "itkAffineTransform.h"
#include "itkTransformFileReader.h"
#include "../inc/itkDeformationFieldGradientTensorImageFilter.h"
#include "itkImageFileWriter.h"
#include "itkDisplacementFieldTransform.h"
#include "JacobianComputation.h"

#include "itkVector.h"
using namespace std;
int main (int argc, char *argv[])
	{
	const unsigned int ImageDimension = 2;
	    GetPot cl (argc, const_cast<char**>(argv));
	    if( cl.size() == 1 || cl.search (2,"--help","-h") )
	    {
	        std::cout << "Not Enough Arguments" << std::endl;
	        std::cout << "Generate the Gradient Table" << std::endl;
	        std::cout << "Usage:  return -1" << std::endl;
	    }

	    std::cout << "Reading Table" << std::endl;

	    const string defField_n = cl.follow("NoFile",1 , "-d");
	    const string maskImage_n = cl.follow("NoFile",1, "-m");
	    const string grad_n = cl.follow("NoFile",1, "-g");



	    typedef itk::Vector<double, 3> VectorType;
	    typedef itk::Image<VectorType, 3> VectorImageType;
	    typedef itk::ImageFileReader<VectorImageType> VectorReaderType;

	    typedef std::vector<VectorType> GradientListType;
	    GradientListType GradientList;
	    std::ifstream fileg(grad_n);
	    int numOfGrads =0;
	    fileg >> numOfGrads;

	    for (int i=0; i <numOfGrads; i++)
	    {
	    	VectorType g;
	    	fileg >> g[0]; fileg >> g[1]; fileg >> g[2];
	    	GradientList.push_back(g);
	    }



	    typedef itk::Image<float, 3> ImageType;
	    typedef itk::ImageFileReader<ImageType> ReaderType;
	    ReaderType::Pointer reader = ReaderType::New();

	    reader->SetFileName(maskImage_n);
	    reader->Update();
	    ImageType::Pointer maskImage = reader->GetOutput();

	    VectorReaderType::Pointer vectorReader = VectorReaderType::New();
	    vectorReader->SetFileName(defField_n.c_str());
	    vectorReader->Update();

	    VectorImageType::Pointer defField = vectorReader->GetOutput();

	    JacobianComputation jacoComputation;
	    jacoComputation.ReadDefField(defField);
	    jacoComputation.ReadMaskImage(maskImage);
	    jacoComputation.ReadGradientList(GradientList);
	    ImageType::Pointer jacDet = jacoComputation.Compute();

	    typedef itk::ImageFileWriter<ImageType> WriterType;
	    WriterType::Pointer writer = WriterType::New();
	    writer->SetFileName("JacobianDet.nii.gz");
	    writer->SetInput(jacDet);
	    writer->Update();

	return 0;
}

