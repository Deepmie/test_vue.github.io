const input=document.getElementsByTagName('input');
const imger=document.getElementById('image1');
const excel1=document.getElementById('excel1');
    const button=document.getElementsByTagName('button')[0];
     button.onclick=()=>{
        excel1.innerHTML="<div class='back'><span id='d'></span></div><span id='t'>loading...</span>"
        let data=`'test_size'=${input[0].value}&'time_limit'=${input[1].value}`
        const xhr=new XMLHttpRequest();
        xhr.open('POST','http://127.0.0.1:8080/s')
        xhr.setRequestHeader('Content-Type','application/x-www-form-urlencoded');
        xhr.send(data);
        xhr.onreadystatechange=()=>{

            if(xhr.readyState===4){
                if(xhr.status>=200&&xhr.status<300){
                    // console.log(xhr.response);
                    response=xhr.response;
                    res_array=response.split('**');
                    res_data=res_array[1];
                    res_data2=res_array[2];
                    opr_array=oprData(res_data);
                    opr_array2=oprData(res_data2);
                    // console.log(res_data);
                    // // opr_res_data=oprData(res_data);
                    // data_array=res_data.split('\n');
                    // data_array.shift();
                    // // console.log(data_array);
                    // let tarray=new Array().fill(0);
                    // number=data_array.indexOf('\r');
                    // // console.log(data_array);
                    // // console.log(number);
                    // for(let i=0;i<number;i++){
                    //     strdata=data_array[i];
                    //     strdata2=data_array[i+number+1];
                    //     strdata3=data_array[i+2*(number+1)];
                    //     new_data=strdata.slice(1,strdata.length-2)+strdata2.slice(1,strdata2.length-2)+strdata3.slice(1,strdata3.length-2);
                    //     // console.log(strdata3);
                    //     tarray.push(new_data)
                    //     // new_data=(data_array[i].substr(1,data_array[i].length-2)+data_array[i+9].substr(1,data_array[i].length-2)+data_array[i+18].substr(1,data_array[i].length-2));
                    //     // tarray.push(new_data);
                    //     // tarray[i]+=data_array[i].slice(1).s+data_array[i+9].slice(1)+data_array[i+18].slice(1);  
                    //     }
                    //     // console.log(tarray);
                    //     let newData=new Array();
                    //     for(let i=0;i<tarray.length;i++){
                    //         iarray=tarray[i].split(' ');
                    //         while(iarray.includes('')){iarray.splice(iarray.indexOf(''),1)};
                    //         // console.log(iarray);
                    //         newData.push(iarray);
                    //     }
                    // console.log(newData);
                    // opr_res_data=newData;
                    // console.log(opr_res_data);
                    // let opr_array=new Array().fill(0);
                    // for(let i=1;i<opr_res_data.length;i++){
                    //     let obj=new Object();
                    //     for(let j=0;j<opr_res_data[i].length;j++){
                    //         obj[opr_res_data[0][j]]=opr_res_data[i][j];
                    //     };
                    //     opr_array.push(obj);
                    //     // console.log(obj);
                    // };
                    // for(let i=1;i<opr_array.length;i++){
                    //     opr_array[i]['Id']=i;
                    //     // console.log(opr_array[i]['Id']);
                    // };
                    console.log(opr_array);
                    // console.log(obj);
                    const title=document.getElementsByClassName('titled');
                    const t1=title[0];
                    const t2=title[1];
                    const t3=title[2];
                    t1.innerHTML='<h1>Conversion Rate CH4 Forecast Leaderboards</h1>';
                    t2.innerHTML='<h1>Hydrogen Yield Forecast Leaderboards</h1>';
                    t3.innerHTML='<h1>SHAP\'s feature importance analysis diagram</h1>';
                    plotExcel('excel1',opr_array);
                    input[0].style.backgroundColor="white";
                    input[1].style.backgroundColor="white";
                    plotExcel('excel2',opr_array2);
                    imger.innerHTML="<img src='C:\\Users\\dell\\Desktop\\autogluonWEB\\SetParameters\\img\\shap.png' />"
                }else{console.log(404)};
            };
        };
    };
    function oprData(res_data){
        data_array=res_data.split('\n');
                    data_array.shift();
                    // console.log(data_array);
                    let tarray=new Array().fill(0);
                    number=data_array.indexOf('\r');
                    // console.log(data_array);
                    console.log(number);
                    for(let i=0;i<number;i++){
                        strdata=data_array[i];
                        strdata2=data_array[i+number+1];
                        strdata3=data_array[i+2*(number+1)];
                        new_data=strdata.slice(1,strdata.length-2)+strdata2.slice(1,strdata2.length-2)+strdata3.slice(1,strdata3.length-2);
                        // console.log(strdata3);
                        tarray.push(new_data)
                        // new_data=(data_array[i].substr(1,data_array[i].length-2)+data_array[i+9].substr(1,data_array[i].length-2)+data_array[i+18].substr(1,data_array[i].length-2));
                        // tarray.push(new_data);
                        // tarray[i]+=data_array[i].slice(1).s+data_array[i+9].slice(1)+data_array[i+18].slice(1);  
                        }
                        // console.log(tarray);
                        let newData=new Array();
                        for(let i=0;i<tarray.length;i++){
                            iarray=tarray[i].split(' ');
                            while(iarray.includes('')){iarray.splice(iarray.indexOf(''),1)};
                            // console.log(iarray);
                            if(i>=11){iarray.shift()};
                            newData.push(iarray);
                        }
                    console.log(newData);
                    opr_res_data=newData;
                    console.log(opr_res_data);
                    let opr_array=new Array().fill(0);
                    for(let i=1;i<opr_res_data.length;i++){
                        let obj=new Object();
                        for(let j=0;j<opr_res_data[i].length;j++){
                            obj[opr_res_data[0][j]]=opr_res_data[i][j];
                        };
                        opr_array.push(obj);
                        // console.log(obj);
                    };
                    for(let i=0;i<opr_array.length;i++){
                        opr_array[i]['Id']=i+1;
                        // console.log(opr_array[i]['Id']);
                    };
                    return opr_array;
    }

    function plotExcel(cssName,Data){
    //set attribute
    const SetWidth=103.5;

    let table = new Tabulator(`#${cssName}`, {//绑定css中的名
    data:Data,           //load row data from array
    layout:"fitColumns",      //fit columns to width of table
    responsiveLayout:"hide",  //hide columns that don't fit on the table
	height:390,
    addRowPos:"top",          //when adding a new row, add it to the top of the table
    history:true,             //allow undo and redo actions on the table
    pagination:"local",       //paginate the data
    paginationSize:50,         //allow 7 rows per page of data
    paginationCounter:"rows", //display count of paginated rows in footer
    movableColumns:true,      //allow column order to be changed
    initialSort:[             //set the initial sort order of the data
        // {column:"id", dir:"asc"},
    ],
    columnDefaults:{
        tooltip:true,         //show tool tips on cells
    },
    columns:[                 //define the table columns
        //formatter:设置样式,
        //hozAlign:设置文字以及柱状图等所处位置
        //width：表格一栏的宽度
        //field：应用属性
        {title:"Id", field:"Id", width:70,hozAlign:"center",sorter:"number"},
        {title:"model", field:"model", width:SetWidth,hozAlign:"center",sorter:"number"},
        {title:"can_infer", field:"can_infer", width:SetWidth,hozAlign:"center",sorter:"number",formatter:"tickCross"},
        {title:"fit_order", field:"fit_order", width:SetWidth,hozAlign:"center",sorter:"number",formatter:"star"},
        {title:"fit_time", field:"fit_time", width:SetWidth,hozAlign:"center",sorter:"number"},
        {title:"fit_time_marginal", field:"fit_time_marginal", width:SetWidth,hozAlign:"center",sorter:"number"},
        {title:"pred_time_val", field:"pred_time_val", width:SetWidth,hozAlign:"center",sorter:"number"},
        {title:"pred_time_val_marginal", field:"pred_time_val_marginal", width:SetWidth,hozAlign:"center",sorter:"number"},
        {title:"score_val", field:"score_val", width:SetWidth,hozAlign:"center",sorter:"number"},
        {title:"stack_level", field:"stack_level", width:SetWidth,hozAlign:"center",sorter:"number"},
    ],
});
}
