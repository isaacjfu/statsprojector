import './App.css';
import {useState,useEffect} from 'react';
import Searchbar from './components/searchbar.js'
import PlayerInfo from './components/playerinfo.js'
const backendURL = 'http://127.0.0.1:5000/'
function App() {
  const [players,setPlayers] = useState([])
  const [selectedPlayer, setSelectedPlayer] = useState("")
  const [playerData , setPlayerData] = useState([])

  useEffect ( () => {
    const playerFetch = async () => {
      try{
        const res = await fetch(backendURL)
        // Check if the response is ok (status code 200-299)
        if (!res.ok) {
          throw new Error(`HTTP error! Status: ${res.status}`);
        }
        const data = await res.json()
        setPlayers(data)
        console.log("in main" + players)
      } catch (error) {
        console.error(error)
      }
    }
    playerFetch()
  }, [])

  useEffect ( () => {
    const playerFetch = async() => {
      try{
        const json_data = JSON.stringify({"name" : selectedPlayer})
        const res = await fetch(backendURL + 'getPlayer', {
          method: "POST",
          headers : {"Content-Type" : "application/json",
            "Accept" : 'application/json'
          },
          body: json_data
        })
        const data = await res.json()
        console.log(data)
        setPlayerData(data)
      } catch (error){
        console.error(error)
      }
    }
    if (selectedPlayer != ""){
      playerFetch()
    }
  }, [selectedPlayer])

  return (
    <div className="App">
      <span>Choose a Player</span>
      <Searchbar players = {players} setSelectedPlayer = {setSelectedPlayer} />
      <PlayerInfo playerData = {playerData} />
    </div>
  );
}

export default App;
