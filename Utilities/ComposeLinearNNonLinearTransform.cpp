/*
 * ComposeLinearNNonLinearTransform.cpp
 *
 *  Created on: Aug 28, 2015
 *      Author: vgupta
 */

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkCompositeTransform.h"
#include "itkVector.h"
#include "../GetPot/GetPot"
#include "itkTransformFileReader.h"
#include "itkTransformFactoryBase.h"


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
    const string Linear_trans_n = cl.follow("NoFile", 1, "-t");

    itk::TransformFactoryBase::RegisterDefaultTransforms();

  #if (ITK_VERSION_MAJOR == 4 && ITK_VERSION_MINOR >= 5) || ITK_VERSION_MAJOR > 4
    itk::TransformFileReaderTemplate<float>::Pointer reader =
      itk::TransformFileReaderTemplate<float>::New();
  #else
    itk::TransformFileReader::Pointer writer = itk::TransformFileReader::New();
  #endif
    reader->SetFileName(Linear_trans_n);
    reader->Update();

    std::cout << *(reader->GetTransformList()->begin()) << std::endl;


  return 0;
}
