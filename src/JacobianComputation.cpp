/*
 * JacobianComputation.cpp
 *
 *  Created on: Aug 24, 2015
 *      Author: vgupta
 */
#include "JacobianComputation.h"
#include "vnl/algo/vnl_svd.h"
#include "CopyImage.h"

void JacobianComputation::ReadDefField(VectorImageType::Pointer defField)
{
	m_DefField = defField;
}

void JacobianComputation::ReadMaskImage(ImageType::Pointer maskImage)
{
	m_MaskImage = maskImage;
}

void JacobianComputation::ReadGradientList(GradientListType gradList)
{
	m_GradList = gradList;
}

JacobianComputation::ImageType::Pointer JacobianComputation::Compute()
{
	typedef itk::ImageRegionIterator<ImageType>  ScalarIterator;
	typedef itk::ImageRegionIterator<VectorImageType> VectorImageIterator;
	typedef itk::NeighborhoodIterator<VectorImageType> VectorNeighborhoodIterator;
	typedef itk::ImageRegionIterator<MatrixImageType> MatrixImageIterator;

	ImageType::Pointer JacobianDet = ImageType::New();
	CopyImage cpImage;
	cpImage.CopyScalarImage(m_MaskImage, JacobianDet);

	MatrixType ZeroM; ZeroM.Fill(0.0);
	MatrixImageType::Pointer matrixImage = MatrixImageType::New();
	matrixImage->SetSpacing(m_MaskImage->GetSpacing());
	matrixImage->SetDirection(m_MaskImage->GetDirection());
	matrixImage->SetOrigin(m_MaskImage->GetOrigin());
	matrixImage->SetRegions(m_MaskImage->GetLargestPossibleRegion());
	matrixImage->Allocate();
	matrixImage->FillBuffer(ZeroM);

	ImageType::SpacingType spacing;
	spacing = m_MaskImage->GetSpacing();
	VectorImageType::SizeType radius;
	radius.Fill(1.0);
	VectorNeighborhoodIterator itDef(radius, m_DefField, m_DefField->GetLargestPossibleRegion());
	ScalarIterator itMask(m_MaskImage, m_MaskImage->GetLargestPossibleRegion());
	ScalarIterator itJacDet(JacobianDet, JacobianDet->GetLargestPossibleRegion());
//	MatrixImageIterator itJac(matrixImage, matrixImage->GetLargestPossibleRegion());

	for (itMask.GoToBegin(), itDef.GoToBegin(), itJacDet.GoToBegin();
			!itMask.IsAtEnd(), !itDef.IsAtEnd(), !itJacDet.IsAtEnd();
				++itMask, ++itDef, ++itJacDet)
	{
		if (itMask.Get() != 0)
		{
			VectorType delU = itDef.GetNext(0) - itDef.GetPrevious(0);
			VectorType delV = itDef.GetNext(1) - itDef.GetPrevious(1);
			VectorType delW = itDef.GetNext(2) - itDef.GetPrevious(2);

			MatrixType J; J.Fill(0.0);
			J(0,0) = delU[0]/(2*spacing[0]); // delu/delx
			J(0,1) = delV[0]/(2*spacing[1]); // delu/dely
			J(0,2) = delW[0]/(2*spacing[2]); // delw/delz

			J(1,0) = delU[1]/(2*spacing[0]); // delv/delx
			J(1,1) = delV[1]/(2*spacing[1]); // delV/dely
			J(1,2) = delW[1]/(2*spacing[2]); //delV/delZ

			J(2,0) = delU[2]/(2*spacing[0]); // delW/delx
			J(2,1) = delV[2]/(2*spacing[1]);
			J(2,2) = delW[2]/(2*spacing[2]);

			vnl_matrix<RealType> _J = J.GetVnlMatrix();
			vnl_matrix<RealType> U,V,W;
			U.set_size(3,3); V.set_size(3,3); W.set_size(3,3);

			vnl_svd<RealType> svd(_J);
			U = svd.U(); V = svd.V(); W = svd.W();

			vnl_matrix<RealType> R,S;
			R=U*V.transpose();

			RealType JacDet;
			RealType a,b,c,d,e,f,g,h,i;
			a=J(0,0); b=J(0,1); c=J(0,2);
			d=J(1,0); e=J(1,1); f=J(1,2);
			g=J(2,0); h=J(2,1); i=J(2,2);

			JacDet = a*e*i + b*f*g + c*d*h - c*e*g - b*d*i - a*f*h;
			itJacDet.Set(JacDet);
//			itJac.Set(J);
			matrixImage->SetPixel(itMask.GetIndex(), R);


		}
	}

	int numOfGrads = m_GradList.size();
	MatrixImageIterator itJac(matrixImage, matrixImage->GetLargestPossibleRegion());

	typedef itk::ImageFileWriter<VectorImageType> VectorImageWriter;

	for (int i =0; i < numOfGrads; i++)
	{
		VectorType ZeroV; ZeroV.Fill(0.0);
		VectorImageType::Pointer GradientImage = VectorImageType::New();
		GradientImage->SetOrigin(m_MaskImage->GetOrigin());
		GradientImage->SetDirection( m_MaskImage->GetDirection());
		GradientImage->SetOrigin(m_MaskImage->GetOrigin());
		GradientImage->SetRegions(m_MaskImage->GetLargestPossibleRegion());
		GradientImage->Allocate();
		GradientImage->FillBuffer(ZeroV);

		VectorImageIterator itGrad(GradientImage, GradientImage->GetLargestPossibleRegion());

		for (itMask.GoToBegin(), itJac.GoToBegin(), itGrad.GoToBegin(); !itMask.IsAtEnd(), !itJac.IsAtEnd(), !itGrad.IsAtEnd();
					++itMask, ++itJac, ++itGrad)
		{
			if (itMask.Get() != 0)
			{
				itGrad.Set(itJac.Get()*m_GradList[i]);
			}
		}

		std::ostringstream c;
		c<< i;

		std::string _C_str;
		_C_str=c.str() ;

		std::string tempName;
		tempName = "Gradient_" + _C_str + ".nii.gz";

		VectorImageWriter::Pointer writer = VectorImageWriter::New();
		writer->SetFileName(tempName);
		writer->SetInput(GradientImage);
		writer->Update();


	}



	return JacobianDet;
}
