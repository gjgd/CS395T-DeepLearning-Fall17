#include<iostream>
#include<vector>
#include<string>
#include<fstream>
#include<map>
using namespace std;

#ifndef SEP_BY_GENDER
#define SEP_BY_GENDER false
#endif
enum
{
	TRAIN = 0,
	VALID = 1,
};

#ifndef TYPE
#define TYPE 0 
#endif

void createClassDir(std::map<std::string, std::vector<std::string>> map, std::string type, std::string postFix)
{
	std::string postFixString = (postFix != "") ? postFix + "/" : "";
	for(std::map<std::string, std::vector<std::string>>::iterator it = map.begin(); it != map.end(); it++)
	{
		std::string year = it->first;
		std::vector<std::string> images =  it->second;
	
		std::string dirCreateStr = "mkdir keras_yearbook/" + type + "/" + postFixString + year;
		system(dirCreateStr.c_str());
	
		for(int i = 0; i < images.size(); i++)
		{
			std::string cpImgStr = "cp " + type + "/" + images[i] + " " + "keras_yearbook/" + type + "/" + postFixString +  year;
			system(cpImgStr.c_str());
		}
	}
}

int main(int argc, char* argv[])
{
	std::string line;
	std::string filename = TYPE ? "yearbook_valid.txt" : "yearbook_train.txt";
	std::string type = TYPE ? "valid" : "train";
	std::string delimiter = "\t";

	std::map<std::string, std::vector<std::string>> yearImMap;
	std::map<std::string, std::vector<std::string>> yearImMapMale;
	std::map<std::string, std::vector<std::string>> yearImMapFemale;

	ifstream myfile(filename);

	if(myfile.is_open())
	{
		while(getline(myfile, line))
		{
			std::string image = line.substr(0, line.find(delimiter));
			std::string year = line.substr(line.find(delimiter) + 1);
			if(!SEP_BY_GENDER)
			{
				yearImMap[year].push_back(image);			
			}
			else
			{
				std::string delim = "/";
				std::string gender = image.substr(0, line.find(delim));
				if(gender == "M")
					yearImMapMale[year].push_back(image);
				else
					yearImMapFemale[year].push_back(image);
			}

		}
		myfile.close();
	}

	std::string dirCreateStr = "mkdir -p keras_yearbook";
	system(dirCreateStr.c_str());

	dirCreateStr = "mkdir -p keras_yearbook/" + type; 
	system(dirCreateStr.c_str());

	if(!SEP_BY_GENDER)
	{
		createClassDir(yearImMap, type, "");
	}
	else
	{
		dirCreateStr = "mkdir -p keras_yearbook/" + type + "/M";
		system(dirCreateStr.c_str());

		dirCreateStr = "mkdir -p keras_yearbook/" + type + "/F";
		system(dirCreateStr.c_str());

		createClassDir(yearImMapMale, type, "M");
		createClassDir(yearImMapFemale, type, "F");
	}

	return 0;
}	
