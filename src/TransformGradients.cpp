#include "TransformGradients.h"
#include "itkDeformationFieldGradientTensorImageFilter.h"

void TransformGradients::ReadMaskImage(ScalarFloatImageType::Pointer image)
{
	m_MaskImage = image;
}

void TransformGradients::ReadDeformationField(VectorImageType::Pointer defImage)
{
	m_defField = defImage;
}

void TransformGradients::ReadGradients(GradientListType list)
{
	m_GradList = list;
}

void TransformGradients::ComputeGradients()
{
	//Jacobian Filter
	JacobianFilterType::Pointer jacobianFilter = JacobianFilterType::New();
	jacobianFilter->SetInput(m_defField);
	jacobianFilter->SetCalculateJacobian(true);
	jacobianFilter->SetUseImageSpacing(true);
	jacobianFilter->SetOrder(1);
	jacobianFilter->SetUseCenteredDifference(true);
	jacobianFilter->Update();
	
	MatrixImageType::Pointer JacobianImage = jacobianFilter->GetOutput();
	int numOfGrads = m_GradList.size();
	
	ScalarFloatIterator itMask(m_MaskImage, m_MaskImage->GetLargestPossibleRegion());
	MatrixImageIterator itJac(JacobianImage, JacobianImage->GetLargestPossibleRegion());	

	for (int i=0; i < numOfGrads; i++)
	{
	  VectorType ZeroD; ZeroD.Fill(0.0);
	  VectorImageType::Pointer gradientImage = VectorImageType::New();
	  gradientImage->SetDirection(m_MaskImage->GetDirection());
	  gradientImage->SetSpacing(m_MaskImage->GetSpacing());
	  gradientImage->SetOrigin(m_MaskImage->GetOrigin());
	  gradientImage->SetRegions(m_MaskImage->GetLargestPossibleRegion());
	  gradientImage->Allocate();
	  gradientImage->FillBuffer(ZeroD);
	
	 VectorIterator itGrad(gradientImage, gradientImage->GetLargestPossibleRegion());
	 for (itMask.GoToBegin(), itJac.GoToBegin(), itGrad.GoToBegin();
	 !itMask.IsAtEnd(), !itJac.IsAtEnd(), !itGrad.IsAtEnd(); 
	++itMask, ++itJac, ++itGrad)
	{
		if (itMask.Get() !=0)
		{
		 vnl_matrix<double> J = itJac.Get().GetVnlMatrix();
		 vnl_matrix<double> U,V,W;
			U.set_size(3,3); V.set_size(3,3); W.set_size(3,3);
		
		vnl_svd<double> svd(J);
		U = svd.U(); V = svd.V(); W = svd.W();

		vnl_matrix<double> R,S;
		R=U*V.transpose();
		
		vnl_vector<double> GRot;
		GRot = R*m_GradList[i].GetVnlVector();

		VectorType test; test.SetVnlVector(GRot);
		itGrad.Set(test);
	
		}
	}
		
	m_GradientImages.push_back(gradientImage);	
	
	std::cout << "Transform Gradients.... " << i << std::endl;		

	}
}

TransformGradients::VectorImageListType TransformGradients::GetGradientImages()
{
	return m_GradientImages;
}
