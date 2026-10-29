#include "strata/program/igc_output_diagnostic.hpp"
#include <iostream>
#include <stdexcept>
int main(int argc,char** argv)try{
    if(argc!=3 && argc!=4)throw std::runtime_error("artifact fixture [prefix-i32-file]");
    strata::program::IgcOutputDiagnostic diagnostic;std::string error;
    if(!diagnostic.load_host(argv[1],argv[2],error))throw std::runtime_error(error);
    bool matched=false;strata::igc::ExactResult selected;
    if(argc==4){
        auto bytes=std::filesystem::file_size(argv[3]);
        if(bytes%4 || bytes<4 || bytes>262144*4)throw std::runtime_error("prefix bytes");
        std::ifstream input(argv[3],std::ios::binary);std::vector<int32_t> ids(static_cast<std::size_t>(bytes/4));
        input.read(reinterpret_cast<char*>(ids.data()),static_cast<std::streamsize>(bytes));if(!input)throw std::runtime_error("prefix read");
        std::vector<int64_t> prefix(ids.begin(),ids.end());
        if(!diagnostic.select(prefix,selected,matched,error))throw std::runtime_error(error);
    }
    std::cout<<"{\"entries\":"<<diagnostic.entries()<<",\"matched\":"<<(matched?"true":"false")
             <<",\"runtime_requests\":"<<(diagnostic.runtime_requests()?"true":"false")<<"}\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
