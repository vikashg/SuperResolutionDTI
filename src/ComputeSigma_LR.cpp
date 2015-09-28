#include "ComputeSigma_LR.h"

void ComputeSigma_LR::ReadDWIList(ImageListType listImage)
{
	m_DWIList = listImage;
//	std::cout << "DWILIst " << std::endl;
}

void ComputeSigma_LR::ReadMaskImage_HR(ScalarImageType::Pointer maskImage)
{
	m_maskImage_HR = maskImage;
//	std::cout << "MaskImage HR " << std::endl;
}

void ComputeSigma_LR::ReadGradientList(VectorImageListType gradList)
{
	m_GradList = gradList;
//	std::cout << "GradientList Read " << std::endl;
}

void ComputeSigma_LR::ReadTensorImage(TensorImageType::Pointer tensorImage)
{
	m_dt = tensorImage;
//	std::cout << "TensorRead " << std::endl;
}

void ComputeSigma_LR::ReadLRImage(ScalarImageType::Pointer image)
{
	m_LRImage = image;
}

void ComputeSigma_LR::ReadBVal(RealType bval)
{
	m_BVal = bval;
//	std::cout << "BVal Read " << std::endl;
}

void ComputeSigma_LR::ReadB0Image_HR(ScalarImageType::Pointer image)
{
	m_B0ImageHR = image;
//	std::cout << "B0Image_HR read " << std::endl;
}

void ComputeSigma_LR::ReadB0Image_LR(ScalarImageType::Pointer image)
{
	m_B0ImageLR = image;
//	std::cout << "B0Image_HR read " << std::endl;
}

void ComputeSigma_LR::ReadMapMatrix(vnl_sparse_matrix<float> map)
{
	m_MapHR2LR = map;
//	std::cout << "map Read " << std::endl;
}


vnl_vector<ComputeSigma_LR::RealType> ComputeSigma_LR::ComputeAttenuation()
{

	int numOfImages = m_DWIList.size();
	int numOfGrads  = m_GradList.size();
	
	TensorUtilities tensUtilities;
	CopyImage cpImage;

	
	vnl_vector<RealType> Sigma; Sigma.set_size(numOfImages);

	for (int i=0; i < numOfGrads ; i++)
	{
	  ScalarImageType::Pointer Atten_im = ScalarImageType::New();
	  cpImage.CopyScalarImage(m_maskImage_HR, Atten_im);

				std::cout << i << std::endl;

	  ScalarImageIterator itAtten(Atten_im, Atten_im->GetLargestPossibleRegion());
	  TensorImageIterator itTens(m_dt, m_dt->GetLargestPossibleRegion());
	  VectorImageIterator itGrad(m_GradList[i], m_GradList[i]->GetLargestPossibleRegion());
	  ScalarImageIterator itMask(m_maskImage_HR, m_maskImage_HR->GetLargestPossibleRegion());
	  ScalarImageIterator itB0(m_B0ImageHR, m_B0ImageHR->GetLargestPossibleRegion());
	  
	 SubtractImageFilterType::Pointer subImageFilter = SubtractImageFilterType::New();
	 StatisticsImageFilterType::Pointer statisticImageFilter = StatisticsImageFilterType::New();	
	
	  MaskImageFilterType::Pointer maskImageFilter = MaskImageFilterType::New();
	for ( itTens.GoToBegin(), itGrad.GoToBegin(), itMask.GoToBegin(), itAtten.GoToBegin(), itB0.GoToBegin();
	  !itTens.IsAtEnd(), !itGrad.IsAtEnd(), !itMask.IsAtEnd(), !itAtten.IsAtEnd(), !itB0.IsAtEnd();
	 ++itTens, ++itGrad, ++itMask, ++itAtten, ++itB0)
	{
		if (itMask.Get() != 0)
		{
				RealType Atten_i_val;
				vnl_vector<double> temp_g = itGrad.Get().GetVnlVector();
				vnl_vector<RealType> g_i; g_i.set_size(3);
				vnl_copy(temp_g, g_i);
				
				vnl_matrix<float> g_mat_i;
				g_mat_i.set_size(3,1);
				g_mat_i.set_column(0,g_i);


				DiffusionTensorType D = itTens.Get();
				MatrixType D_mat;
				D_mat.set_size(3,3);
				D_mat = tensUtilities.ConvertDT2Mat(D);

				MatrixType temp; temp.set_size(1,1);
				temp = g_mat_i.transpose()*D_mat*g_mat_i;

				Atten_i_val = exp(-m_BVal*temp(0,0))*itB0.Get();
				itAtten.Set(Atten_i_val);		
		}
	}

//		std::cout << "Computed Atten " << std::endl;	
		
//		std::cout << m_MapHR2LR.rows() << " " << m_MapHR2LR.cols() << std::endl;
		ComposeImageFilter composeFilter;
		composeFilter.GetHRImage(Atten_im);
		composeFilter.GetLRImage(m_LRImage);
		composeFilter.ReadMatrix(m_MapHR2LR);

		ScalarImageType::Pointer atten_im_LR = composeFilter.ComposeIt();

//		std::cout << "Composed " << std::endl;
		subImageFilter->SetInput1(m_DWIList[i]);
		subImageFilter->SetInput2(atten_im_LR);
		subImageFilter->Update();

		ScalarImageType::Pointer diffImage = subImageFilter->GetOutput();
		diffImage->DisconnectPipeline();		

//		std::cout << "DiffImage " << std::endl;
				
		DivideByImageFilterType::Pointer divideByImageFilter = DivideByImageFilterType::New();
		
		divideByImageFilter->SetInput1(diffImage);
		divideByImageFilter->SetInput2(m_B0ImageLR);
		divideByImageFilter->Update();

		ScalarImageType::Pointer fracImage = divideByImageFilter->GetOutput();
		fracImage->DisconnectPipeline();
		
//		std::cout << "FracImage " << std::endl;
			
		maskImageFilter->SetMaskImage(m_LRImage);
		maskImageFilter->SetInput(fracImage);
		maskImageFilter->Update();

		ScalarImageType::Pointer maskedFracImage = maskImageFilter->GetOutput();

	//	BinaryThresholdImageFilterType::Pointer binaryThreholdImageFilter = BinaryThresholdImageFilterType::New();
		ThresholdImageFilterType::Pointer binaryThreholdImageFilter = ThresholdImageFilterType::New();
		binaryThreholdImageFilter->SetInput(maskedFracImage);
		binaryThreholdImageFilter->ThresholdOutside(-1.1, 1.1);
		binaryThreholdImageFilter->SetOutsideValue(0.0);
		binaryThreholdImageFilter->Update();
		ScalarImageType::Pointer thres_Image = binaryThreholdImageFilter->GetOutput();
		thres_Image->DisconnectPipeline();

/*		 BinaryThresholdImageFilterType::Pointer thresholdFilter= BinaryThresholdImageFilterType::New();
  thresholdFilter->SetInput(maskedFracImage);
  thresholdFilter->SetLowerThreshold(10);
  thresholdFilter->Update();
*
	ScalarImageType::Pointer img = thresholdFilter->GetOutput();*/
		statisticImageFilter->SetInput(thres_Image);
		statisticImageFilter->Update();
		RealType sig = statisticImageFilter->GetSigma();

		Sigma.put(i, sig);

		typedef itk::ImageFileWriter<ScalarImageType> ScalarWriterType;
		ScalarWriterType::Pointer scalarWriter = ScalarWriterType::New();
		std::ostringstream c ;
		c<< i;
		std::string _C_str;
		_C_str = c.str();
		std::string tempName, tempName1;
		tempName = "Image_" + _C_str + ".nii.gz";
		tempName1 = "FracImage_" + _C_str + ".nii.gz";
		scalarWriter->SetFileName(tempName);
		scalarWriter->SetInput(diffImage);
		scalarWriter->Update();

//		ScalarWriterType::Pointer scalarWriter1 = ScalarWriterType::New();
//		scalarWriter1->SetFileName(tempName1);
//		scalarWriter1->SetInput(img);
//		scalarWriter1->Update();
		
	
	}

		return Sigma;
}

