/*
 * TransformGradientTable_1.cpp
 *
 *  Created on: Aug 25, 2015
 *      Author: vgupta
 */

#include "itkImage.h"
#include "itkDeformationFieldGradientTensorImageFilter.h"
#include "../GetPot/GetPot"
#include "itkVector.h"
#include "itkImageFileReader.h"
#include "itkImageRegionIterator.h"
#include "vnl/algo/vnl_svd.h"
#include "itkHDF5ImageIO.h"
#include "itkImageFileWriter.h"

using namespace std;
int main(int argc, char *argv[])
{
    GetPot cl (argc, const_cast<char**>(argv));
    if( cl.size() == 1 || cl.search (2,"--help","-h") )
	    {
	        std::cout << "Not Enough Arguments" << std::endl;
	        std::cout << "Generate the Gradient Table" << std::endl;
	        std::cout << "Usage:  return -1" << std::endl;
	    }

    const string defField_n = cl.follow("NoFile",1 , "-d");
    const string maskImage_n = cl.follow("NoFile",1, "-m");
    const string grad_n = cl.follow("NoFile",1, "-g");


    typedef itk::Vector<double, 3> VectorType;
    typedef itk::Image<VectorType, 3> VectorImageType;
    typedef itk::ImageFileReader<VectorImageType> VectorReaderType;
    typedef itk::ImageFileWriter<VectorImageType> VectorImageWriter;

    typedef itk::Image<float, 3> ImageType;
    typedef itk::ImageFileReader<ImageType> ReaderType;

    ReaderType::Pointer reader = ReaderType::New();
    reader->SetFileName(maskImage_n);
    reader->Update();

    ImageType::Pointer maskImage = reader->GetOutput();

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

    VectorReaderType::Pointer readerDef= VectorReaderType::New();
    readerDef->SetFileName(defField_n);
    readerDef->Update();
    VectorImageType::Pointer defField = readerDef->GetOutput();

    typedef itk::DeformationFieldGradientTensorImageFilter<VectorImageType, double> JacobianFilterType;
    JacobianFilterType::Pointer jacobianFilter = JacobianFilterType::New();
    jacobianFilter->SetInput(defField);
    jacobianFilter->SetCalculateJacobian(true);
    jacobianFilter->SetUseImageSpacing(true);
    jacobianFilter->SetOrder(1);
    jacobianFilter->SetUseCenteredDifference(true);
    jacobianFilter->Update();

//    std::cout << defField->GetOrigin() << std::endl;

    VectorImageType::IndexType testIndex;
    testIndex[0]=60; testIndex[1]=111; testIndex[2]=128;

//    std::cout << jacobianFilter->GetOutput()->GetPixel(testIndex) << std::endl;

    typedef itk::Matrix<double, 3, 3> MatrixType;
    typedef itk::Image<MatrixType, 3> MatrixImageType;
    MatrixImageType::Pointer JacobianImage = jacobianFilter->GetOutput();

    typedef itk::ImageRegionIterator<MatrixImageType> JacobianIterator;
    JacobianIterator itJac(JacobianImage, JacobianImage->GetLargestPossibleRegion());

    typedef itk::ImageRegionIterator<ImageType> ImageIterator;
    ImageIterator itMask(maskImage, maskImage->GetLargestPossibleRegion());

    typedef itk::ImageRegionIterator<VectorImageType> VectorIterator;

    for (int i=0; i < numOfGrads; i++)
    {
    	VectorType ZeroD; ZeroD.Fill(0.0);
    	VectorImageType::Pointer gradientImage = VectorImageType::New();
    	gradientImage->SetDirection(maskImage->GetDirection());
    	gradientImage->SetSpacing(maskImage->GetSpacing());
    	gradientImage->SetOrigin(maskImage->GetOrigin());
    	gradientImage->SetRegions(maskImage->GetLargestPossibleRegion());
    	gradientImage->Allocate();
    	gradientImage->FillBuffer(ZeroD);

    	VectorIterator itGrad(gradientImage, gradientImage->GetLargestPossibleRegion());

    for(itMask.GoToBegin(), itJac.GoToBegin(), itGrad.GoToBegin();
    		!itMask.IsAtEnd(), !itJac.IsAtEnd(), !itGrad.IsAtEnd();
    		++itMask, ++itJac, ++itGrad)
    {
    	if (itMask.Get() != 0)
    	{
    		vnl_matrix<double> Id; Id.set_identity(); Id.set_size(3,3);
    		vnl_matrix<double> J = itJac.Get().GetVnlMatrix();
    		vnl_matrix<double> U,V,W;
			U.set_size(3,3); V.set_size(3,3); W.set_size(3,3);



			vnl_svd<double> svd(J);
			U=svd.U(); V= svd.V(); W = svd.W();

			vnl_matrix<double> R,S;
			R=U*V.transpose();


			vnl_vector<double> GRot;
			GRot = R*GradientList[i].GetVnlVector();

			VectorType test; test.SetVnlVector(GRot);
			itGrad.Set(test);
    	}
    }
    std::ostringstream c;
    c<< i;
    std::cout << "Image " << i << "done " <<  GradientList[i] << std::endl;
    std::string _C_str;
    _C_str=c.str() ;

    std::string tempName;
    tempName = "Gradient_" + _C_str + ".nii.gz";

    VectorImageWriter::Pointer writer = VectorImageWriter::New();
    writer->SetFileName(tempName);
    writer->SetInput(gradientImage);
    writer->Update();


    }
    return 0;


}


