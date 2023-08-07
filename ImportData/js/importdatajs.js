const readexcel=document.getElementById('readexcel');
const showMatrix=document.getElementById('showMatrix');
const showBox1_text=document.getElementById('showBox1-Text');
const showBox2_text=document.getElementById('showBox2-Text');
const showBox1=document.getElementById('showBox1');
const showBox2=document.getElementById('showBox2');
const content=document.getElementById('content1');
    readexcel.onchange=()=>{
        file=readexcel.files[0];
        const fileReader=new FileReader();
        content.style.height="3800px";
        fileReader.readAsText(file);
        fileReader.onload=function(){
            // console.log(this.result);
            result=this.result;
            //数据清理等
            while(true){
                last_result=result;
                result=result.replace('\r','');
                if(last_result===result){break};
            };
            array=result.split('\n');
            //进行数据的转化
            array.every((value,index,array)=>{array[index]=array[index].split(',');return true});
            // console.log(array);
            //服务箱型图数据
            let array_box=[...array];
            array_box=array_box.slice(1,array.length-1);
            array_box.every((value,index,array)=>{
                value.every((value_i,index_i,array_i)=>{array_i[index_i]=Number(value_i);return true});
                array[index]=value;
                return true;
            })
            // console.log(array_box);
            //数组转为obj
            let Data=new Array(array.length-2).fill(0);
            Data=Data.map((dvalue,dindex,darray)=>{
                let number=0;
                let obj=new Object();
                obj["id"]=dindex+1;
                for(iarray of array[0]){obj[iarray]=Number(array[dindex+1][number++])};
                return obj;
            })
            console.log(Data);
            //将Data利用表格展现出来
            plotExcel("showExcel",Data);
            //将Data利用箱型图展现出来
            const boxData=transpose(array_box);
            const boxData1=boxData.slice(0,4);
            const boxData2=boxData.slice(4,8);
            // const boxData1=boxData.slice(0,4);
            const label=Object.values(array[0]);
            plotBox("showBox1",boxData1,label.slice(0,4),1);
            showBox1_text.innerHTML="<h1>Box plot of the first four features</h1>"
            plotBox("showBox2",boxData2,label.slice(4,8),2);
            showBox2_text.innerHTML="<h1>Box plot of the last four features</h1>"
            
            console.log(array_box);
            ToData(array_box);
        };
    };
function plotExcel(cssName,Data){
    //set attribute
    const SetWidth=113;

    let table = new Tabulator(`#${cssName}`, {//绑定css中的名
    data:Data,           //load row data from array
    layout:"fitColumns",      //fit columns to width of table
    responsiveLayout:"hide",  //hide columns that don't fit on the table
	height:1420,
    addRowPos:"top",          //when adding a new row, add it to the top of the table
    history:true,             //allow undo and redo actions on the table
    pagination:"local",       //paginate the data
    paginationSize:50,         //allow 7 rows per page of data
    paginationCounter:"rows", //display count of paginated rows in footer
    movableColumns:true,      //allow column order to be changed
    initialSort:[             //set the initial sort order of the data
        {column:"id", dir:"asc"},
    ],
    columnDefaults:{
        tooltip:true,         //show tool tips on cells
    },
    columns:[                 //define the table columns
        //formatter:设置样式,
        //hozAlign:设置文字以及柱状图等所处位置
        //width：表格一栏的宽度
        //field：应用属性
        {title:"ID", field:"id", width:70,hozAlign:"center",sorter:"number"},
        {title:"Temperature", field:"Temperature", width:SetWidth,hozAlign:"center",sorter:"number"},
        {title:"Water Flow Rate", field:"Water Flow Rate", width:SetWidth,hozAlign:"center",sorter:"number"},
        {title:"Volumetric Flow Rate CH4", field:"Volumetric Flow Rate CH4", width:SetWidth,hozAlign:"center",sorter:"number"},
        {title:"Volumetric Flow Rate CO2", field:"Volumetric Flow Rate CO2", width:SetWidth,hozAlign:"center",sorter:"number"},
        {title:"Space Velocity", field:"Space Velocity", width:SetWidth,hozAlign:"center",sorter:"number"},
        {title:"Pressure", field:"Pressure", width:SetWidth,hozAlign:"center",sorter:"number"},
        {title:"Conversion Rate CH4", field:"Conversion Rate CH4", width:SetWidth,hozAlign:"center",sorter:"number"},
        {title:"Hydrogen Yield", field:"Hydrogen Yield", width:SetWidth,hozAlign:"center",sorter:"number"},

    ],
});
}
const deleted=document.getElementById('delete');
const showExcel=document.getElementById('showExcel');
const readExcel=document.getElementById('readexcel');
deleted.onclick=()=>{
	showExcel.innerHTML="You can upload a standard form with headers that should contain:<br /><ur><li>Temperature</li><li>Water Flow Rate</li><li>Volumetric Flow Rate CH4</li><li>Volumetric Flow Rate CO2</li><li>Pressure</li><li>Conversion Rate CH4</li><li>Hydrogen Yield</li></ur>";
	showExcel.style.border="0px solid red";
	showExcel.style="border-radius: 50px;padding:10px;margin-top:0px;background-color: rgba(22, 212, 219, 0.432);font-size:23px;font-family: 'Times new Roman';text-indent: 50px;line-height: 25px;"
	readExcel.value="";
    showBox1_text.innerHTML="";
    showBox2_text.innerHTML="";
    showBox1.innerHTML="";
    showBox2.innerHTML="";
    showMatrix.innerHTML="";
}
readExcel.onclick=()=>{
	showExcel.style="border:0px;border-radius:0px;font-size:20px;line-height:19px;text-indent:0px;padding:10px;margin-top:0px;background-color: white;font-family: 'Times new Roman';"

}
function plotBox(cssName,myData,label,index){
    // console.log(myData);
    // console.log(label)
    let myChartDom=document.getElementById(cssName);
    let chart = echarts.init(myChartDom)  // 初始化画布节点
    let options={
        title:[
            {text:"Boxed"+String(index),left:'center'},
            {text: 'upper: Q3 + 1.5 * IQR \nlower: Q1 - 1.5 * IQR',
            borderColor: '#999',
            borderWidth: 1,
            textStyle: {fontWeight: 'normal',fontSize: 14,lineHeight: 20 },
            left: '10%',
            bottom:'10%'
            }
        ],
        dataset:[{
            // source:[
            //     [850, 740, 900, 1070, 930, 850, 950, 980, 980, 880, 1000, 980, 930, 650, 760, 810, 1000, 1000, 960, 960],
            //     [960, 940, 960, 940, 880, 800, 850, 880, 900, 840, 830, 790, 810, 880, 880, 830, 800, 790, 760, 800],
            //     [880, 880, 880, 860, 720, 720, 620, 860, 970, 950, 880, 910, 850, 870, 840, 840, 850, 840, 840, 840],
            //     [890, 810, 810, 820, 800, 770, 760, 740, 750, 760, 910, 920, 890, 860, 880, 720, 840, 850, 850, 780],
            //     [890, 840, 780, 810, 760, 810, 790, 810, 820, 850, 870, 870, 810, 740, 810, 940, 950, 800, 810, 870]
            // ]
            source:myData
        },{
            transform: {
                type: 'boxplot',
                config: { itemNameFormatter: label[Number('{value}')] }
            }
        },{
            fromDatasetIndex: 1,
            fromTransformResult: 1
        }],

        tooltip:{
        trigger: 'item',
        axisPointer: {
            type: 'shadow'
        }
        },
        grid: {
            left: '10%',
            right: '10%',
            bottom: '30%'
        },
        xAxis: {
            type: 'category',
            boundaryGap: true,
            nameGap: 30,
            splitArea: {
                show: false
            },
            splitLine: {
                show: false
            },
            axisLabel:{
                formatter:(params)=>{
                    return label[params];
                }
            }
        },
        yAxis: {
            type: 'value',
            name: 'km/s minus 299,000',
            splitArea: {
                show: true
            }
        },
        series: [
            {
                name: 'boxplot',
                type: 'boxplot',
                datasetIndex: 1
            },
            {
                name: 'outlier',
                type: 'scatter',
                datasetIndex: 2
            }
        ]
    };
    chart.setOption(options)   // 将配置项在画布上画出来
}
function transpose(array1){
    let array2=new Array(array1[0].length).fill(0);
    array2.every((value,index,array)=>{
        array[index]=new Array(array1.length);
        return true;
    });
    for(var i=0;i<array1.length;i++){
        for(var j=0;j<array1[0].length;j++){
            array2[j][i]=array1[i][j];
        };
    };
    return array2;
}
function ToData(data){
    const xhr=new XMLHttpRequest();
    xhr.open('POST','http://127.0.0.1:8080/server',true);
    xhr.setRequestHeader('Content-Type','application/x-www-form-urlencoded');
    // xhr.setRequestHeader('Content-Type','application/json');
    const DATA={
        "value":data
    };
    // xhr.send(JSON.stringify(DATA));
    xhr.send("value="+JSON.stringify(data));
    xhr.onreadystatechange=()=>{
        if(xhr.readyState===4){
            if(xhr.status>=200&&xhr.status<300){
                console.log(xhr.response);
            }else{console.log(404)};
        };
    };
    showMatrix.innerHTML="<h1>Pearson correlation matrix</h1><img src='./js/matrix-figure.jpg' width='50%'/>"
}
