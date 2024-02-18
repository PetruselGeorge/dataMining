First thing first you need to install the ElasticSearch,

and after the installation you will go to the config file in elasticsearch folder where you will find a yaml file called elastic search.

After opening, go to the BEGIN SECURITY AUTO CONFIGURATION section and set everything to false.

Then open the PowerShell or equivalent and navigate to the bin file and run ./elasticsearch.bat

after this you will open the main.py firstly change the openai key with yours(located in the first lines of code),

and in the main() you will have to remove the two '#' .I commented those function because you will need

to use them only the first time when you're running the programme so the wiki get indexed.

The change the host to your ipv4 address port:9200
