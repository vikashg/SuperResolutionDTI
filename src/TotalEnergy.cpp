/*
 * TotalEnergy.cxx
 *
 *  Created on: Aug 11, 2015
 *      Author: vgupta
 */

#include "TotalEnergy.h"

void TotalEnergy::ReadGradientImageList(VectorImageListType GradList)
{
	m_GradList = GradList;
}

void TotalEnergy::ReadPredImageList(ImageListType imageList)
{
	m_predList = imageList;
}

void TotalEnergy::ReadLRMaskImage(ScalarImageType::Pointer image)
{
	m_LRMaskImage = image;
}

void TotalEnergy::ReadHRMaskImage(ScalarImageType::Pointer image)
{
	m_HRMaskImage = image;
}
void TotalEnergy::ReadB0Image(ScalarImageType::Pointer image)
{
	m_B0Image = image;
}

void TotalEnergy::ReadMapMatrixHR2LR(SparseMatrixType map)
{
	m_MapHR2LR = map;
}

void TotalEnergy::ReadObsImageList(ImageListType DWIList)
{
	m_DWIList = DWIList;
}


void TotalEnergy::ReadBValue(RealType bVal)
{
	m_BVal = bVal;
}

void TotalEnergy::ReadKappa(RealType kappa)
{
	m_Kappa = kappa;
}

void TotalEnergy::ReadSigma(vnl_vector<RealType> sigma)
{
	m_Sigma = sigma;

}


void TotalEnergy::SetFracFlag(bool flag)
{
	m_FracFlag = flag;
}



TotalEnergy::RealType TotalEnergy::GaussianNoise_SR(TensorImageType::Pointer logTensorImage)
{
	RealType gaussEnergy =0;

	int numOfImages = m_DWIList.size();
	
	ImageListType predImageList_LR;
	// Convert PredList to LR;
	for (int i=0; i < numOfImages; i++)
	{
	ComposeImageFilter composeFilter;	
	composeFilter.GetHRImage(m_predList[i]);
	composeFilter.GetLRImage(m_LRMaskImage);
	composeFilter.ReadMatrix(m_MapHR2LR);
	
	ScalarImageType::Pointer image_i = composeFilter.ComposeIt();
	predImageList_LR.push_back(image_i);

	}

	
	ScalarImageType::IndexType testIndex;
	testIndex[0]=59; testIndex[1]=58; testIndex[2]=27;

	ScalarImageIterator itLRMask(m_LRMaskImage, m_LRMaskImage->GetLargestPossibleRegion());
	ScalarImageIterator itB0(m_B0Image, m_B0Image->GetLargestPossibleRegion());


	for (int i=0; i < numOfImages; i++)
	{
	
	      std::ostringstream num_con;
              num_con << i;
              std::string result  = num_con.str();
              std::string tempName2 = "PredImage_LR_" + result + ".nii.gz";
	

	ScalarWriterType::Pointer writer = ScalarWriterType::New();
	writer->SetFileName(tempName2);
	writer->SetInput(predImageList_LR[i]);
	writer->Update();

	ScalarImageIterator it(predImageList_LR[i], predImageList_LR[i]->GetLargestPossibleRegion());
	
	for(it.GoToBegin(), itLRMask.GoToBegin(); !it.IsAtEnd(), !itLRMask.IsAtEnd(); ++it, ++itLRMask)
	{
		if (itLRMask.Get() != 0)
		{
			if ( (isnan(it.Get() ) == 1) || (isinf(it.Get()) ==1 ) )
			{
				//check the tensors etc for now a patch
				it.Set(0.0);
			}
		}
	
	} 


	}

	ScalarImageType::Pointer GaussEnergy = ScalarImageType::New();
	CopyImage cpImage;
	cpImage.CopyScalarImage(m_DWIList[0], GaussEnergy);

	for (int i =0; i < numOfImages; i++)		
	{
		ScalarImageIterator itPred(predImageList_LR[i],predImageList_LR[i]->GetLargestPossibleRegion());
		ScalarImageIterator itObs(m_DWIList[i], m_DWIList[i]->GetLargestPossibleRegion());
			
		ScalarImageIterator itGauss(GaussEnergy, GaussEnergy->GetLargestPossibleRegion());

		
	if (m_FracFlag == 1)
	{

	 	for (itGauss.GoToBegin(), itLRMask.GoToBegin(), itPred.GoToBegin(), itObs.GoToBegin(), itB0.Get(); 
		!itPred.IsAtEnd(), !itGauss.IsAtEnd(), !itObs.IsAtEnd(), !itLRMask.IsAtEnd(), !itB0.IsAtEnd(); 
		++itPred, ++itObs, ++itLRMask, ++itB0, ++itGauss)
		{
		 if (itLRMask.Get() !=0)
		 {

			RealType temp_diff =0;
			temp_diff = (itPred.Get() - itObs.Get());
			
			if (itLRMask.GetIndex() == testIndex)
			{
				std::cout << "Pred " << itPred.Get() << std::endl;
			}	 
	
			RealType deno =1;
			RealType energy_vox =0;
			if(itB0.Get() > 100)
			{
				deno = itB0.Get()*m_Sigma[i];
				energy_vox = temp_diff/deno;				
			}
			else
			{
			   energy_vox =0;
			}
			
			gaussEnergy += energy_vox*energy_vox;
			RealType temp = itGauss.Get();
			itGauss.Set(temp+energy_vox*energy_vox);
			
		 }		
		} 
	}
	else
	{
		  
	 	for (itLRMask.GoToBegin(), itPred.GoToBegin(), itObs.GoToBegin(); 
		!itPred.IsAtEnd(), !itObs.IsAtEnd(), !itLRMask.IsAtEnd(); 
		++itPred, ++itObs, ++itLRMask)
		{
			RealType temp_diff = 0;
			temp_diff = (itPred.Get() - itObs.Get());
			RealType  deno =1;
			deno = m_Sigma[i];

			RealType energy_vox =0;
			energy_vox = temp_diff/deno;
			
			gaussEnergy += energy_vox*energy_vox;
		
		}
	}  
	}
	
	typedef itk::ImageFileWriter<ScalarImageType> ScalarImageWriterType;
	ScalarImageWriterType::Pointer scalarImageWriter = ScalarImageWriterType::New();
	scalarImageWriter->SetFileName("GaussEnergy.nii.gz");
	scalarImageWriter->SetInput(GaussEnergy);
	scalarImageWriter->Update();



	return gaussEnergy;
}


TotalEnergy::RealType TotalEnergy::RegularizationEnergy(TensorImageType::Pointer logTensorImage)
{
	JointTensorEstimation jTestimation;
	jTestimation.ReadHRMask(m_HRMaskImage);
	ScalarImageType::Pointer gradMagTensorImage = jTestimation.GradientLogMagTensorImage(logTensorImage) ;

	//std::cout << "GradMagTensorImage " << std::endl;

	ScalarImageIterator itMask(m_HRMaskImage, m_HRMaskImage->GetLargestPossibleRegion());
	ScalarImageIterator itGrad(gradMagTensorImage, gradMagTensorImage->GetLargestPossibleRegion());

	RealType Total =0;
	for (itMask.GoToBegin(), itGrad.GoToBegin(); !itMask.IsAtEnd(), !itGrad.IsAtEnd();
			++itMask, ++itGrad)
	{
		if (itMask.Get() != 0)
		{ RealType regTerm, temp;
		temp = itGrad.Get()/m_Kappa;
		temp = 2*sqrt(1+ temp*temp) -2;
		Total = Total + temp;
		}
	}

	return Total;
}


