const input=document.getElementsByTagName('input');
const button=document.getElementsByTagName('button')[0];
const show_input=document.getElementsByClassName('show-input');
button.onclick=()=>{
        // console.log(input[0].value,input[1].value,input[2].value,input[3].value,input[4].value,input[5].value,
        // input[6].value,input[7].value,input[8].value,input[9].value,input[10].value,input[11].value);
        //偶数分为一组==>上界
        //奇数分为一组==>下界,
        data='n-iter='+String(input[0].value)+'&lb='+String([input[1].value,input[3].value,input[5].value,input[7].value,input[9].value,input[11].value,])+'&ub='+String([input[2].value,input[4].value,input[6].value,input[8].value,input[10].value,input[12].value,])
        const xhr=new XMLHttpRequest();
        xhr.open('POST','http://127.0.0.1:8080/stas')
        xhr.setRequestHeader('Content-Type','application/x-www-form-urlencoded');
        xhr.send(data);
        xhr.onreadystatechange=()=>{
            if(xhr.readyState===4){
                if(xhr.status>=200&&xhr.status<300){
                    // console.log(xhr.response);
                    data=oprData(xhr.response);
                    for(let i=0;i<data.length;i++){
                        show_input[i].value=data[i];
                    }
                    console.log(data);
                }else{console.log(404)};
            };
        };
    }
    function oprData(fileData){
    dataArray=fileData.split('\n');
    // console.log(dataArray)
    // name_str=dataArray[1]+dataArray[4]+dataArray[7];
    // console.log(fileData)
    let name_str=dataArray[2].substr(1)+dataArray[5].substr(1)+dataArray[8].substr(1);
    // console.log(name_str);
    name_str_old='';
    while(name_str!=name_str_old){
        name_str_old=name_str;
        name_str=name_str.replace('\\','');
        name_str=name_str.replace('\r','');
        // name_str.replace('\\','');
    };
    name_str=name_str.split(' ');
    const name_array=new Array();
    while(name_str.includes('')){name_str.splice(name_str.indexOf(''),1)};
    for(let i=0;i<name_str.length;i++){name_array.push(Number(name_str[i]))};
    return(name_array)
};