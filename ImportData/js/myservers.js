const express=require('express');
var bodyParser = require('body-parser');
const app=express();
app.use(bodyParser.urlencoded({ extended: false }));
app.use(bodyParser.json());

app.post('/server',(request,response)=>{
    response.setHeader('Access-Control-Allow-Origin','*');
    let accept_data=request.body;
    // response.send(typeof(accept_data));
    accept_data_json=JSON.stringify(Object.values(accept_data));
    // response.send((accept_data_json));
    // response.send(JSON.parse(Object.values(accept_data)));
    // let act_data=JSON.parse(Object.values(accept_data));
    // let act_data=(Object.keys(accept_data)[0]);
    // response.send(act_data);
    // response.send(accept_data);
    let exec=require('child_process').exec;
    execCmd();
    function execCmd(){
        exec(`python responseData.py ${accept_data_json}`,(error,stdout,stderr)=>{
            if(error){console.log(404)};
            // console.log(stdout);
            response.send(stdout);
            // console.log(stderr);
        });
    };
    
});
app.post('/s',(request,response)=>{
    console.log(accept_data_json);
    response.setHeader('Access-Control-Allow-Origin','*');
    // response.send(request.body);
    data=JSON.stringify(request.body);
    let exec=require('child_process').exec;
    execCmd();
    function execCmd(){// 
        exec(`cd C:\\Users\\dell\\Desktop\\autogluonWEB && conda activate PyTorch && python ./test/trytocal.py ${data} ${accept_data_json}`,(error,stdout,stderr)=>{
            if(error){console.log(404)};
            response.send(stdout);
        })
        // exec(`python ./pythonNodejs/arrangePy/s1.py ${nodeData}`,(error,stdout,stderr)=>{
        //     if(error){console.log(404)};
        //     console.log(stdout);
        //     // console.log(stderr);
        // })
    }
})
app.post('/stas',(request,response)=>{
    console.log(accept_data_json);
    response.setHeader('Access-Control-Allow-Origin','*');
    // response.send(request.body);
    // response.send(request.body);
    let res_data=request.body;
    // response.send(res_data);
    function predata(res_data,dataName){
        res_data_name=res_data[dataName];
        res_name_array=res_data_name.split(',');
        let res_data_array=new Array().fill(0); 
        for(let i=0;i<res_name_array.length;i++){
            res_data_array.push(Number(res_name_array[i]));
        };
        return res_data_array
    }
    res_data_new1=predata(res_data,'lb');
    res_data_new2=predata(res_data,'ub');
    res_data_new3=Number(res_data['n-iter']);
    // const objtt=JSON.stringify({'lb':res_data_new1,'ub':res_data_new2});
    // response.send(res_data)
    let exec=require('child_process').exec;
    execCmd();
    function execCmd(){//
        exec(`cd C:\\Users\\dell\\Desktop\\autogluonWEB && conda activate PyTorch && python ./MOPSOoptimization/python/PStt.py ${accept_data_json} ${res_data_new3} ${res_data_new1} ${res_data_new2}`,(error,stdout,stderr)=>{
            if(error){console.log(404)};
            response.send(stdout);
        })
        // exec(`python ./pythonNodejs/arrangePy/s1.py ${nodeData}`,(error,stdout,stderr)=>{
        //     if(error){6console.log(404)};
        //     console.log(stdout);
        //     // console.log(stderr);
        // })
    }
})
app.listen(8080,()=>{
    console.log('running at http://127.0.0.1:8080');
});
