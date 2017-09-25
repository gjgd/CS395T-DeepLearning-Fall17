#include<iostream>
#include<vector>
#include<string>
#include<fstream>
#include<map>
#include<algorithm>
using namespace std;

#ifndef SEP_BY_GENDER
#define SEP_BY_GENDER false
#endif
enum
{
	TRAIN = 0,
	VALID = 1,
};

void createClassDir(std::vector<std::string> years, std::map<std::string, std::vector<std::string>> map, std::string type, std::string postFix)
{
	std::string postFixString = (postFix != "") ? postFix + "/" : "";

	for(unsigned i = 0; i < years.size(); i++)
	{
		std::string dirCreateStr = "mkdir keras_yearbook/" + type + "/" + postFixString + years[i];
		system(dirCreateStr.c_str());
	}

	for(std::map<std::string, std::vector<std::string>>::iterator it = map.begin(); it != map.end(); it++)
	{
		std::string year = it->first;
		std::vector<std::string> images =  it->second;
	
	
		for(int i = 0; i < images.size(); i++)
		{
			std::string cpImgStr = "cp " + type + "/" + images[i] + " " + "keras_yearbook/" + type + "/" + postFixString +  year;
			system(cpImgStr.c_str());
		}
	}
}

void processData(std::string filename, std::string type, std::vector<std::string>& years)
{
	std::string line;
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
				if(type == "train") years.push_back(year);
			}
			else
			{
				std::string delim = "/";
				std::string gender = image.substr(0, line.find(delim));
				if(gender == "M")
				{
					if(type == "train") years.push_back(year);
					yearImMapMale[year].push_back(image);
				}
				else
				{
					if(type == "train") years.push_back(year);
					yearImMapFemale[year].push_back(image);
				}
			}

		}
		myfile.close();
	}

	std::string dirCreateStr = "mkdir -p keras_yearbook";
	system(dirCreateStr.c_str());

	dirCreateStr = "mkdir -p keras_yearbook/" + type; 
	system(dirCreateStr.c_str());

	std::sort(years.begin(), years.end());
	auto last = std::unique(years.begin(), years.end());
	years.erase(last, years.end());

	if(!SEP_BY_GENDER)
	{
		createClassDir(years, yearImMap, type, "");
	}
	else
	{
		dirCreateStr = "mkdir -p keras_yearbook/" + type + "/M";
		system(dirCreateStr.c_str());

		dirCreateStr = "mkdir -p keras_yearbook/" + type + "/F";
		system(dirCreateStr.c_str());

		createClassDir(years, yearImMapMale, type, "M");
		createClassDir(years, yearImMapFemale, type, "F");
	}
}

int main(int argc, char* argv[])
{
	std::vector<std::string> years;

	processData("yearbook_train.txt", "train", years);
	processData("yearbook_valid.txt", "valid", years);

	return 0;
}	
